import os
import sqlite3
import sys

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DB = "backend/hotspot_validator.db"


def cek_token():
    if not TOKEN or "masukkan_" in TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN belum diisi di .env")


def cmd_id():
    cek_token()
    try:
        data = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=15).json()
    except Exception as e:
        raise SystemExit(f"Gagal terhubung ke Telegram ({type(e).__name__}). Cek internet.")
    if not data.get("ok"):
        raise SystemExit(f"Telegram menolak: {data.get('description')}. Cek token di .env")
    ditemukan = {}
    for u in data.get("result", []):
        pesan = u.get("message") or u.get("edited_message") or {}
        chat = pesan.get("chat")
        if chat:
            ditemukan[chat["id"]] = chat.get("first_name") or chat.get("title") or chat.get("username") or "?"
    if not ditemukan:
        print("Belum ada pesan. Buka bot-mu di Telegram, kirim /start, lalu jalankan lagi.")
    for cid, nama in ditemukan.items():
        print(f"chat_id = {cid}   nama = {nama}")


def cmd_daftar(chat_id, nama):
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO subscriptions (chat_id, nama, wilayah, aktif) VALUES (?, ?, 'SEMUA', 1) "
        "ON CONFLICT(chat_id) DO UPDATE SET aktif = 1, nama = excluded.nama",
        (str(chat_id), nama),
    )
    conn.commit()
    print("Pelanggan aktif sekarang:")
    for baris in conn.execute("SELECT chat_id, nama, aktif FROM subscriptions"):
        print(" ", tuple(baris))
    conn.close()


def cmd_uji():
    cek_token()
    from backend.app.database import get_connection
    from backend.app.main import send_telegram_alert

    contoh = {
        "wilayah": "Pulang Pisau (Gambut), Kalteng",
        "skor_akhir": 82.5,
        "level": "Kemungkinan besar api",
        "satelit": "VIIRS_SNPP",
        "frp": 41.2,
        "lat": -3.0833,
        "lon": 114.3833,
        "faktor_utama": '[{"faktor": "Energi Radiasi Api (FRP)", "nilai_fitur": 41.2, "dampak": "+1.8 (meningkatkan risiko)"}]',
    }
    conn = get_connection()
    send_telegram_alert(contoh, conn)
    conn.close()


if __name__ == "__main__":
    perintah = sys.argv[1] if len(sys.argv) > 1 else ""
    if perintah == "id":
        cmd_id()
    elif perintah == "daftar" and len(sys.argv) >= 3:
        cmd_daftar(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "Saya")
    elif perintah == "uji":
        cmd_uji()
    else:
        print("Pemakaian:\n  python telegram_tes.py id\n  python telegram_tes.py daftar CHAT_ID NAMA\n  python telegram_tes.py uji")