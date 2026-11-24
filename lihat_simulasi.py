import sqlite3

conn = sqlite3.connect("backend/hotspot_validator.db")

total = conn.execute("SELECT COUNT(*) FROM hotspots").fetchone()[0]
print("Total titik panas:", total)

print("\nJumlah per jenis (wilayah):")
for wilayah, jumlah in conn.execute(
    "SELECT wilayah, COUNT(*) FROM hotspots GROUP BY wilayah ORDER BY COUNT(*) DESC"
):
    print(f"  {jumlah:5d}  {wilayah}")

print("\n10 contoh titik yang bukan data NASA dan bukan laporan warga:")
contoh = conn.execute(
    "SELECT id, wilayah, satelit, waktu_akuisisi FROM hotspots "
    "WHERE wilayah != 'Kalimantan (VIIRS)' AND satelit != 'LAPORAN_WARGA' LIMIT 10"
).fetchall()
for baris in contoh:
    print(" ", baris)

conn.close()