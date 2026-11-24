import base64
import hashlib
import hmac
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

ITERASI = 600_000
MASA_BERLAKU_DETIK = 8 * 3600  # sesi login berlaku 8 jam
SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(32).hex()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERASI)
    return f"pbkdf2_sha256${ITERASI}${salt.hex()}${dk.hex()}"


def verify_password(password: str, tersimpan: str) -> bool:
    try:
        algo, iterasi, salt_hex, hash_hex = tersimpan.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterasi)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(teks: str) -> bytes:
    return base64.urlsafe_b64decode(teks + "=" * (-len(teks) % 4))


def _tanda_tangan(isi: str) -> str:
    return hmac.new(SECRET_KEY.encode(), isi.encode(), hashlib.sha256).hexdigest()


def create_token(email: str, role: str, nama: str) -> str:
    payload = json.dumps({
        "email": email, "role": role, "nama": nama,
        "exp": int(time.time()) + MASA_BERLAKU_DETIK,
    }).encode()
    isi = _b64(payload)
    return f"{isi}.{_tanda_tangan(isi)}"


def read_token(token: str):
    """Kembalikan data pengguna kalau token sah dan belum kedaluwarsa, selain itu None."""
    try:
        isi, tanda = token.split(".")
        if not hmac.compare_digest(tanda, _tanda_tangan(isi)):
            return None
        data = json.loads(_unb64(isi))
        if data["exp"] < time.time():
            return None
        return data
    except Exception:
        return None