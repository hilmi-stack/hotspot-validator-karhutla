import getpass
import sys

from backend.app.database import init_db, get_connection
from backend.app.security import hash_password


def main():
    if len(sys.argv) != 2:
        print("Pemakaian: python -m backend.set_password email@contoh.id")
        return

    email = sys.argv[1].strip().lower()
    init_db()
    conn = get_connection()
    user = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if not user:
        print(f"Pengguna {email} tidak ditemukan.")
        conn.close()
        return

    pw1 = getpass.getpass("Password baru (min. 8 karakter): ")
    pw2 = getpass.getpass("Ulangi password: ")
    if pw1 != pw2:
        print("Password tidak sama. Dibatalkan.")
    elif len(pw1) < 8:
        print("Password minimal 8 karakter. Dibatalkan.")
    else:
        conn.execute("UPDATE users SET password_hash = ? WHERE email = ?", (hash_password(pw1), email))
        conn.commit()
        print(f"Password untuk {email} berhasil diatur.")
    conn.close()


if __name__ == "__main__":
    main()