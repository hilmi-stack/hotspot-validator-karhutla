import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "backend/hotspot_validator.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs("backend", exist_ok=True)
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS hotspots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        wilayah TEXT DEFAULT 'Kalimantan Tengah',
        waktu_akuisisi TEXT NOT NULL,
        satelit TEXT DEFAULT 'VIIRS_SNPP',
        confidence REAL DEFAULT 65.0,
        frp REAL DEFAULT 15.0,
        curah_hujan_7h REAL DEFAULT 0.0,
        suhu REAL DEFAULT 32.0,
        kelembapan REAL DEFAULT 65.0,
        kecepatan_angin REAL DEFAULT 8.0,
        jenis_lahan TEXT DEFAULT 'Gambut',
        jarak_permukiman_km REAL DEFAULT 3.5,
        skor_dasar REAL DEFAULT 0.0,
        skor_boost REAL DEFAULT 0.0,
        skor_akhir REAL DEFAULT 0.0,
        level TEXT DEFAULT 'Perlu dicek',
        faktor_utama TEXT DEFAULT '[]',
        versi_model TEXT DEFAULT 'xgb_hotspot_v1',
        notified INTEGER DEFAULT 0,
        dibuat_pada TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kode_laporan TEXT UNIQUE,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        waktu TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        catatan TEXT DEFAULT '',
        kontak TEXT DEFAULT '',
        foto_url TEXT NOT NULL,
        hash_foto TEXT,
        label_ai TEXT DEFAULT 'bukan',
        keyakinan_ai REAL DEFAULT 0.0,
        valid INTEGER DEFAULT 1,
        alasan_tidak_valid TEXT DEFAULT '',
        hotspot_id INTEGER
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS field_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hotspot_id INTEGER NOT NULL,
        hasil TEXT NOT NULL,
        catatan TEXT DEFAULT '',
        foto_url TEXT DEFAULT '',
        petugas TEXT DEFAULT 'Petugas Lapangan',
        waktu TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT UNIQUE,
        nama TEXT DEFAULT 'Relawan',
        wilayah TEXT DEFAULT 'SEMUA',
        aktif INTEGER DEFAULT 1
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ambang_tinggi REAL DEFAULT 70.0,
        ambang_sedang REAL DEFAULT 40.0,
        radius_validasi_km REAL DEFAULT 10.0,
        nilai_boost REAL DEFAULT 15.0,
        versi_model_aktif TEXT DEFAULT 'xgb_hotspot_v1'
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS job_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nama_job TEXT,
        status TEXT,
        jumlah_data INTEGER DEFAULT 0,
        pesan TEXT DEFAULT '',
        waktu TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        nama TEXT DEFAULT 'Petugas',
        role TEXT DEFAULT 'petugas'
    )
    """)

    cur.execute("SELECT COUNT(*) FROM settings")
    if cur.fetchone()[0] == 0:
        cur.execute("""
        INSERT INTO settings (ambang_tinggi, ambang_sedang, radius_validasi_km, nilai_boost, versi_model_aktif)
        VALUES (70.0, 40.0, 10.0, 15.0, 'xgb_hotspot_v1')
        """)

    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO users (email, nama, role) VALUES ('admin@karhutla.id', 'Admin Utama', 'admin')")
        cur.execute("INSERT INTO users (email, nama, role) VALUES ('petugas@karhutla.id', 'Tim Lapangan', 'petugas')")

    conn.commit()
    conn.close()