import os
import shutil
import sqlite3
from datetime import datetime

DB = "backend/hotspot_validator.db"

if not os.path.exists(DB):
    raise SystemExit("Database tidak ditemukan. Jalankan dari folder C:\\Projects\\hotspot-validator")

cadangan = f"backend/cadangan_{datetime.now():%Y%m%d_%H%M%S}.db"
shutil.copy(DB, cadangan)
print("Cadangan dibuat:", cadangan)

conn = sqlite3.connect(DB)
kondisi = """
    wilayah != 'Kalimantan (VIIRS)'
    AND satelit != 'LAPORAN_WARGA'
    AND id NOT IN (SELECT hotspot_id FROM reports WHERE hotspot_id IS NOT NULL)
    AND id NOT IN (SELECT hotspot_id FROM field_checks)
"""
total = conn.execute("SELECT COUNT(*) FROM hotspots").fetchone()[0]
target = conn.execute(f"SELECT COUNT(*) FROM hotspots WHERE {kondisi}").fetchone()[0]
print(f"Total titik: {total} | akan dihapus (simulasi): {target} | tersisa: {total - target}")

if target and input("Ketik 'ya' untuk menghapus: ").strip().lower() == "ya":
    conn.execute(f"DELETE FROM hotspots WHERE {kondisi}")
    conn.commit()
    print("Selesai.")
else:
    print("Dibatalkan, tidak ada yang dihapus.")
conn.close()