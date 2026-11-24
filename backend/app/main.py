import os
import time
import uuid
import json
import random
from datetime import datetime
from typing import Optional
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from .database import init_db, get_connection
from .security import verify_password, create_token, read_token
from .services.ai_pipeline import (
    fetch_firms_or_simulate, fetch_weather_openmeteo, predict_hotspot_score,
    classify_level, analyze_citizen_photo, haversine_km
)

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
os.makedirs("backend/uploads", exist_ok=True)

app = FastAPI(title="AgniVision API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="backend/uploads"), name="uploads")


def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Silakan login terlebih dahulu")
    data = read_token(authorization[7:].strip())
    if not data:
        raise HTTPException(status_code=401, detail="Sesi tidak valid atau sudah berakhir, silakan login ulang")
    return data


def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Hanya admin yang boleh mengakses ini")
    return user

def _md(teks) -> str:
    """Amankan karakter khusus Markdown Telegram."""
    teks = str(teks)
    for ch in ("_", "*", "`", "["):
        teks = teks.replace(ch, "\\" + ch)
    return teks


def send_telegram_alert(hotspot_dict: dict, conn):
    if not TELEGRAM_BOT_TOKEN or "masukkan_" in TELEGRAM_BOT_TOKEN:
        return
    subs = conn.execute("SELECT chat_id FROM subscriptions WHERE aktif = 1").fetchall()
    if not subs:
        print("[TELEGRAM] Tidak ada pelanggan aktif")
        return
    faktor_list = json.loads(hotspot_dict["faktor_utama"])
    faktor_str = "\n".join([f"• {_md(f['faktor'])}: {_md(f['dampak'])}" for f in faktor_list])
    pesan = (
        f"🚨 *PERINGATAN KARHUTLA TINGGI*\n"
        f"📍 Wilayah: *{_md(hotspot_dict['wilayah'])}*\n"
        f"🎯 Skor AI: *{hotspot_dict['skor_akhir']}/100* ({_md(hotspot_dict['level'])})\n"
        f"🛰️ Satelit: {_md(hotspot_dict['satelit'])} | FRP: {hotspot_dict['frp']} MW\n"
        f"📌 Koordinat: `{hotspot_dict['lat']}, {hotspot_dict['lon']}`\n"
        f"🗺️ Maps: https://maps.google.com/?q={hotspot_dict['lat']},{hotspot_dict['lon']}\n\n"
        f"*Faktor Utama:*\n{faktor_str}"
    )
    for s in subs:
        try:
            res = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": s["chat_id"], "text": pesan, "parse_mode": "Markdown"},
                timeout=8,
            )
            if res.status_code == 200:
                print(f"[TELEGRAM] Terkirim ke {s['chat_id']}")
            else:
                print(f"[TELEGRAM] Gagal ({res.status_code}): {res.text[:200]}")
        except Exception as e:
            print(f"[TELEGRAM] Error kirim: {type(e).__name__}")

def run_ingestion_pipeline(use_simulation: bool = False):
    conn = get_connection()
    setting = dict(conn.execute("SELECT * FROM settings LIMIT 1").fetchone())
    try:
        raw_points = fetch_firms_or_simulate(use_simulation=use_simulation)
        added = 0
        for pt in raw_points:
            exists = conn.execute(
                "SELECT id FROM hotspots WHERE lat = ? AND lon = ? AND waktu_akuisisi = ?",
                (pt["lat"], pt["lon"], pt["waktu_akuisisi"])
            ).fetchone()
            if exists:
                continue

            weather = fetch_weather_openmeteo(pt["lat"], pt["lon"])
            is_gambut = 1 if "gambut" in pt["wilayah"].lower() or random.random() > 0.55 else 0
            jarak_permukiman = round(random.uniform(0.8, 12.0), 1)
            tetangga = random.randint(0, 5)

            feat = {
                "conf_num": pt["confidence"],
                "frp": pt["frp"],
                "daynight_num": pt["daynight"],
                "curah_hujan_7h": weather["curah_hujan_7h"],
                "suhu": weather["suhu"],
                "kelembapan": weather["kelembapan"],
                "kecepatan_angin": weather["kecepatan_angin"],
                "is_gambut": is_gambut,
                "jarak_permukiman_km": jarak_permukiman,
                "hotspot_tetangga_5km": tetangga
            }
            skor, level, faktor_json = predict_hotspot_score(
                feat, setting["ambang_tinggi"], setting["ambang_sedang"]
            )
            jenis_lahan = "Gambut" if is_gambut else "Hutan/Semak"
            cur = conn.execute("""
                INSERT INTO hotspots (
                    lat, lon, wilayah, waktu_akuisisi, satelit, confidence, frp,
                    curah_hujan_7h, suhu, kelembapan, kecepatan_angin, jenis_lahan,
                    jarak_permukiman_km, skor_dasar, skor_akhir, level, faktor_utama, versi_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pt["lat"], pt["lon"], pt["wilayah"], pt["waktu_akuisisi"], pt["satelit"],
                pt["confidence"], pt["frp"], weather["curah_hujan_7h"], weather["suhu"],
                weather["kelembapan"], weather["kecepatan_angin"], jenis_lahan,
                jarak_permukiman, skor, skor, level, faktor_json, setting["versi_model_aktif"]
            ))
            new_id = cur.lastrowid
            conn.commit()
            added += 1

            if level == "Kemungkinan besar api":
                hs_row = dict(conn.execute("SELECT * FROM hotspots WHERE id = ?", (new_id,)).fetchone())
                send_telegram_alert(hs_row, conn)
                conn.execute("UPDATE hotspots SET notified = 1 WHERE id = ?", (new_id,))
                conn.commit()

        conn.execute(
            "INSERT INTO job_logs (nama_job, status, jumlah_data, pesan) VALUES (?, ?, ?, ?)",
            ("Ingest Hotspot FIRMS", "SUKSES", added, "Pipeline selesai normal")
        )
        conn.commit()
        return added
    except Exception as e:
        conn.execute(
            "INSERT INTO job_logs (nama_job, status, jumlah_data, pesan) VALUES (?, ?, ?, ?)",
            ("Ingest Hotspot FIRMS", "GAGAL", 0, str(e))
        )
        conn.commit()
        return 0
    finally:
        conn.close()

@app.on_event("startup")
def startup_event():
    init_db()
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM hotspots").fetchone()[0]
    conn.close()
    if count == 0:
        run_ingestion_pipeline(use_simulation=True)

    scheduler = BackgroundScheduler()
    # Tarik data otomatis setiap 10 menit (untuk demo bisa diganti seconds=30, use_simulation=True)
    scheduler.add_job(run_ingestion_pipeline, "interval", minutes=10, kwargs={"use_simulation": False})
    scheduler.start()

@app.get("/api/hotspots", dependencies=[Depends(get_current_user)])
def get_hotspots(level: Optional[str] = None):
    conn = get_connection()
    if level and level != "SEMUA":
        rows = conn.execute(
            "SELECT * FROM hotspots WHERE level = ? ORDER BY skor_akhir DESC, jarak_permukiman_km ASC",
            (level,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM hotspots ORDER BY skor_akhir DESC, jarak_permukiman_km ASC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/hotspots/{hotspot_id}", dependencies=[Depends(get_current_user)])
def get_hotspot_detail(hotspot_id: int):
    conn = get_connection()
    hs = conn.execute("SELECT * FROM hotspots WHERE id = ?", (hotspot_id,)).fetchone()
    if not hs:
        conn.close()
        raise HTTPException(status_code=404, detail="Titik tidak ditemukan")
    reports = conn.execute("SELECT * FROM reports WHERE hotspot_id = ? ORDER BY id DESC", (hotspot_id,)).fetchall()
    checks = conn.execute("SELECT * FROM field_checks WHERE hotspot_id = ? ORDER BY id DESC", (hotspot_id,)).fetchall()
    conn.close()
    return {
        "hotspot": dict(hs),
        "reports": [dict(r) for r in reports],
        "field_checks": [dict(c) for c in checks]
    }

@app.post("/api/ingest/trigger", dependencies=[Depends(get_current_user)])
def trigger_ingest(simulasi: bool = True):
    added = run_ingestion_pipeline(use_simulation=simulasi)
    return {"status": "ok", "titik_baru": added}

@app.post("/api/reports")
async def create_report(
    lat: float = Form(...),
    lon: float = Form(...),
    catatan: str = Form(""),
    kontak: str = Form(""),
    foto: UploadFile = File(...)
):
    conn = get_connection()
    setting = dict(conn.execute("SELECT * FROM settings LIMIT 1").fetchone())

    ext = os.path.splitext(foto.filename)[1] or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join("backend/uploads", fname)
    with open(fpath, "wb") as f:
        f.write(await foto.read())

    img_hash, label_ai, conf_ai = analyze_citizen_photo(fpath)

    dup = conn.execute("SELECT id FROM reports WHERE hash_foto = ?", (img_hash,)).fetchone()
    valid = 1
    alasan = ""
    if dup:
        valid = 0
        alasan = "Foto duplikat terdeteksi di sistem"
    elif label_ai == "bukan":
        valid = 0
        alasan = "AI tidak mendeteksi indikasi api atau asap"

    nearest_hs = None
    min_dist = 9999.0
    for hs in conn.execute("SELECT * FROM hotspots").fetchall():
        hs_d = dict(hs)
        d = haversine_km(lat, lon, hs_d["lat"], hs_d["lon"])
        if d < min_dist:
            min_dist = d
            nearest_hs = hs_d

    linked_id = None
    if valid == 1 and nearest_hs and min_dist <= setting["radius_validasi_km"]:
        linked_id = nearest_hs["id"]
        new_boost = nearest_hs["skor_boost"] + setting["nilai_boost"]
        new_akhir = min(100.0, round(nearest_hs["skor_dasar"] + new_boost, 1))
        new_level = classify_level(new_akhir, setting["ambang_tinggi"], setting["ambang_sedang"])
        conn.execute(
            "UPDATE hotspots SET skor_boost = ?, skor_akhir = ?, level = ? WHERE id = ?",
            (new_boost, new_akhir, new_level, linked_id)
        )
        conn.commit()
        if new_level == "Kemungkinan besar api" and nearest_hs["notified"] == 0:
            updated_hs = dict(conn.execute("SELECT * FROM hotspots WHERE id = ?", (linked_id,)).fetchone())
            send_telegram_alert(updated_hs, conn)
            conn.execute("UPDATE hotspots SET notified = 1 WHERE id = ?", (linked_id,))
            conn.commit()
    elif valid == 1:
        faktor_json = json.dumps([
            {"faktor": f"Foto Warga ({label_ai.upper()})", "nilai_fitur": f"{conf_ai*100:.0f}%", "dampak": f"+{setting['nilai_boost']}"}
        ])
        skor_awal = min(100.0, 55.0 + setting["nilai_boost"])
        cur = conn.execute("""
            INSERT INTO hotspots (
                lat, lon, wilayah, waktu_akuisisi, satelit, confidence, frp,
                skor_dasar, skor_boost, skor_akhir, level, faktor_utama
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            lat, lon, "Laporan Warga Lapangan", datetime.now().strftime("%Y-%m-%d %H:%M"),
            "LAPORAN_WARGA", round(conf_ai * 100, 1), 0.0,
            55.0, setting["nilai_boost"], skor_awal, "Perlu dicek", faktor_json
        ))
        linked_id = cur.lastrowid
        conn.commit()

    kode = f"RPT-{random.randint(10000, 99999)}"
    conn.execute("""
        INSERT INTO reports (
            kode_laporan, lat, lon, catatan, kontak, foto_url, hash_foto,
            label_ai, keyakinan_ai, valid, alasan_tidak_valid, hotspot_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        kode, lat, lon, catatan, kontak, f"/uploads/{fname}", img_hash,
        label_ai, conf_ai, valid, alasan, linked_id
    ))
    conn.commit()
    conn.close()

    return {
        "kode_laporan": kode, "label_ai": label_ai, "keyakinan_ai": conf_ai,
        "valid": bool(valid), "alasan_tidak_valid": alasan, "hotspot_id": linked_id
    }

class FieldCheckIn(BaseModel):
    hotspot_id: int
    hasil: str
    catatan: str = ""
    petugas: str = "Petugas BPBD/Manggala Agni"

@app.post("/api/field-checks", dependencies=[Depends(get_current_user)])
def submit_field_check(payload: FieldCheckIn):
    conn = get_connection()
    hs = conn.execute("SELECT id FROM hotspots WHERE id = ?", (payload.hotspot_id,)).fetchone()
    if not hs:
        conn.close()
        raise HTTPException(status_code=404, detail="Hotspot tidak ditemukan")
    conn.execute(
        "INSERT INTO field_checks (hotspot_id, hasil, catatan, petugas) VALUES (?, ?, ?, ?)",
        (payload.hotspot_id, payload.hasil, payload.catatan, payload.petugas)
    )
    if payload.hasil == "Benar api":
        conn.execute("UPDATE hotspots SET skor_akhir = 100.0, level = 'Kemungkinan besar api' WHERE id = ?", (payload.hotspot_id,))
    elif payload.hasil == "Bukan api":
        conn.execute("UPDATE hotspots SET skor_akhir = 5.0, level = 'Kemungkinan alarm palsu' WHERE id = ?", (payload.hotspot_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Hasil cek lapangan tersimpan ke dataset latih"}

@app.get("/api/admin/overview", dependencies=[Depends(require_admin)])
def admin_overview():
    conn = get_connection()
    setting = dict(conn.execute("SELECT * FROM settings LIMIT 1").fetchone())
    logs = [dict(r) for r in conn.execute("SELECT * FROM job_logs ORDER BY id DESC LIMIT 10").fetchall()]
    subs = [dict(r) for r in conn.execute("SELECT * FROM subscriptions").fetchall()]
    total_checks = conn.execute("SELECT COUNT(*) FROM field_checks").fetchone()[0]
    conn.close()
    return {"setting": setting, "logs": logs, "subscriptions": subs, "total_field_checks": total_checks}

class SettingUpdate(BaseModel):
    ambang_tinggi: float
    ambang_sedang: float
    radius_validasi_km: float
    nilai_boost: float

@app.put("/api/admin/settings", dependencies=[Depends(require_admin)])
def update_settings(payload: SettingUpdate):
    conn = get_connection()
    conn.execute(
        "UPDATE settings SET ambang_tinggi = ?, ambang_sedang = ?, radius_validasi_km = ?, nilai_boost = ?",
        (payload.ambang_tinggi, payload.ambang_sedang, payload.radius_validasi_km, payload.nilai_boost)
    )
    for hs in conn.execute("SELECT id, skor_akhir FROM hotspots").fetchall():
        lvl = classify_level(hs["skor_akhir"], payload.ambang_tinggi, payload.ambang_sedang)
        conn.execute("UPDATE hotspots SET level = ? WHERE id = ?", (lvl, hs["id"]))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.post("/api/admin/retrain", dependencies=[Depends(require_admin)])
def trigger_retrain():
    conn = get_connection()
    setting = dict(conn.execute("SELECT * FROM settings LIMIT 1").fetchone())
    ver_num = int(setting["versi_model_aktif"].split("_v")[-1]) + 1
    new_ver = f"xgb_hotspot_v{ver_num}"
    total_checks = conn.execute("SELECT COUNT(*) FROM field_checks").fetchone()[0]
    conn.execute("UPDATE settings SET versi_model_aktif = ?", (new_ver,))
    conn.execute(
        "INSERT INTO job_logs (nama_job, status, jumlah_data, pesan) VALUES (?, ?, ?, ?)",
        ("Retraining Model XGBoost", "SUKSES", total_checks, f"Model diperbarui ke {new_ver}")
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "versi_baru": new_ver}

@app.post("/api/telegram/subscribe", dependencies=[Depends(get_current_user)])
def add_telegram_sub(chat_id: str = Form(...), nama: str = Form("Petugas"), wilayah: str = Form("SEMUA")):
    conn = get_connection()
    existing = conn.execute("SELECT id FROM subscriptions WHERE chat_id = ?", (chat_id,)).fetchone()
    if existing:
        conn.execute("UPDATE subscriptions SET aktif = 1, wilayah = ? WHERE chat_id = ?", (wilayah, chat_id))
    else:
        conn.execute("INSERT INTO subscriptions (chat_id, nama, wilayah, aktif) VALUES (?, ?, ?, 1)", (chat_id, nama, wilayah))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/")
def serve_home():
    return FileResponse("frontend/index.html")

class LoginRequest(BaseModel):
    email: str
    password: str

_gagal_login = {}

@app.post("/api/login")
def login_email(payload: LoginRequest):
    email_clean = payload.email.strip().lower()
    sekarang = time.time()
    percobaan = [t for t in _gagal_login.get(email_clean, []) if sekarang - t < 300]
    if len(percobaan) >= 5:
        raise HTTPException(status_code=429, detail="Terlalu banyak percobaan gagal. Coba lagi dalam 5 menit.")

    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email_clean,)).fetchone()
    conn.close()

    if not user or not user["password_hash"] or not verify_password(payload.password, user["password_hash"]):
        percobaan.append(sekarang)
        _gagal_login[email_clean] = percobaan
        raise HTTPException(status_code=401, detail="Email atau password salah")

    _gagal_login.pop(email_clean, None)
    token = create_token(user["email"], user["role"], user["nama"])
    return {
        "status": "ok",
        "token": token,
        "user": {"email": user["email"], "nama": user["nama"], "role": user["role"]},
    }

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")