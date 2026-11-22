import os
import json
import math
import random
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import imagehash
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")
XGB_MODEL_PATH = "backend/models_store/xgb_hotspot_v1.json"
YOLO_CUSTOM_PATH = "backend/models_store/yolo_fire_v1.pt"

# Muat model XGBoost & SHAP Explainer
xgb_model = xgb.XGBClassifier()
if os.path.exists(XGB_MODEL_PATH):
    xgb_model.load_model(XGB_MODEL_PATH)
    explainer = shap.TreeExplainer(xgb_model)
else:
    explainer = None

# Muat model YOLOv8
if os.path.exists(YOLO_CUSTOM_PATH):
    yolo_model = YOLO(YOLO_CUSTOM_PATH)
    IS_CUSTOM_YOLO = True
else:
    yolo_model = YOLO("yolov8n.pt")
    IS_CUSTOM_YOLO = False

FEATURE_LABELS = {
    'conf_num': 'Confidence Satelit',
    'frp': 'Energi Radiasi Api (FRP)',
    'daynight_num': 'Waktu Akuisisi Siang',
    'curah_hujan_7h': 'Curah Hujan 7 Hari',
    'suhu': 'Suhu Permukaan',
    'kelembapan': 'Kelembapan Udara',
    'kecepatan_angin': 'Kecepatan Angin',
    'is_gambut': 'Lahan Gambut Rawan',
    'jarak_permukiman_km': 'Jarak ke Permukiman',
    'hotspot_tetangga_5km': 'Klaster Hotspot Sekitar'
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def fetch_weather_openmeteo(lat: float, lon: float):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
            f"&daily=precipitation_sum&past_days=7&forecast_days=1"
        )
        res = requests.get(url, timeout=6)
        if res.status_code == 200:
            data = res.json()
            curr = data.get("current", {})
            rain_list = data.get("daily", {}).get("precipitation_sum", [0])
            rain_7d = sum([r for r in rain_list if r is not None])
            return {
                "suhu": float(curr.get("temperature_2m", 32.0)),
                "kelembapan": float(curr.get("relative_humidity_2m", 65.0)),
                "kecepatan_angin": float(curr.get("wind_speed_10m", 8.0)),
                "curah_hujan_7h": round(float(rain_7d), 1)
            }
    except Exception:
        pass
    return {
        "suhu": round(random.uniform(30.5, 35.5), 1),
        "kelembapan": round(random.uniform(52.0, 78.0), 1),
        "kecepatan_angin": round(random.uniform(4.0, 15.0), 1),
        "curah_hujan_7h": round(random.uniform(0.0, 18.0), 1)
    }

def classify_level(skor: float, ambang_tinggi: float = 70.0, ambang_sedang: float = 40.0):
    if skor >= ambang_tinggi:
        return "Kemungkinan besar api"
    elif skor >= ambang_sedang:
        return "Perlu dicek"
    return "Kemungkinan alarm palsu"

def predict_hotspot_score(features_dict: dict, ambang_tinggi: float = 70.0, ambang_sedang: float = 40.0):
    cols = [
        'conf_num', 'frp', 'daynight_num', 'curah_hujan_7h',
        'suhu', 'kelembapan', 'kecepatan_angin', 'is_gambut',
        'jarak_permukiman_km', 'hotspot_tetangga_5km'
    ]
    X_df = pd.DataFrame([[features_dict[c] for c in cols]], columns=cols)
    
    if explainer is not None:
        prob = float(xgb_model.predict_proba(X_df)[0][1])
        skor = round(prob * 100, 1)
        shap_vals = explainer.shap_values(X_df)[0]
        idx_top = np.argsort(np.abs(shap_vals))[::-1][:3]
        faktor_top = []
        for idx in idx_top:
            col_name = cols[idx]
            val_kontribusi = round(float(shap_vals[idx]), 2)
            arah = "meningkatkan risiko" if val_kontribusi >= 0 else "menurunkan risiko"
            faktor_top.append({
                "faktor": FEATURE_LABELS.get(col_name, col_name),
                "nilai_fitur": features_dict[col_name],
                "dampak": f"{'+' if val_kontribusi>=0 else ''}{val_kontribusi} ({arah})"
            })
    else:
        skor = round(min(98.0, features_dict['conf_num'] * 0.8 + features_dict['frp'] * 0.3), 1)
        faktor_top = [{"faktor": "Confidence Satelit", "nilai_fitur": features_dict['conf_num'], "dampak": "+0.8"}]

    level = classify_level(skor, ambang_tinggi, ambang_sedang)
    return skor, level, json.dumps(faktor_top)

def fetch_firms_or_simulate(use_simulation: bool = False):
    if not use_simulation and FIRMS_MAP_KEY and "masukkan_" not in FIRMS_MAP_KEY:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/108.5,-4.2,119.0,4.3/1"
        try:
            res = requests.get(url, timeout=12)
            if res.status_code == 200 and "latitude" in res.text:
                from io import StringIO
                df = pd.read_csv(StringIO(res.text)).head(25)
                records = []
                for _, row in df.iterrows():
                    conf_raw = str(row.get('confidence', 'n')).lower()
                    conf_num = 90.0 if conf_raw == 'h' else (65.0 if conf_raw == 'n' else 35.0)
                    records.append({
                        "lat": float(row['latitude']),
                        "lon": float(row['longitude']),
                        "wilayah": "Kalimantan (VIIRS)",
                        "waktu_akuisisi": f"{row['acq_date']} {str(row['acq_time']).zfill(4)}",
                        "satelit": "VIIRS_SNPP",
                        "confidence": conf_num,
                        "frp": float(row.get('frp', 15.0)),
                        "daynight": 1 if row.get('daynight', 'D') == 'D' else 0
                    })
                if records:
                    return records
        except Exception:
            pass

    sample_locations = [
        (-2.2136, 113.9108, "Palangka Raya, Kalteng"),
        (-2.5333, 112.9500, "Kotawaringin Timur, Kalteng"),
        (-3.0833, 114.3833, "Pulang Pisau (Gambut), Kalteng"),
        (-1.6833, 113.4167, "Gunung Mas, Kalteng"),
        (-0.0263, 109.3425, "Kubu Raya, Kalbar"),
        (-1.8333, 109.9833, "Ketapang, Kalbar"),
        (-2.9667, 115.1500, "Hulu Sungai Selatan, Kalsel"),
        (-0.5022, 117.1536, "Kutai Kartanegara, Kaltim")
    ]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    records = []
    for lat, lon, wil in random.sample(sample_locations, 5):
        records.append({
            "lat": round(lat + random.uniform(-0.08, 0.08), 5),
            "lon": round(lon + random.uniform(-0.08, 0.08), 5),
            "wilayah": wil,
            "waktu_akuisisi": now_str,
            "satelit": random.choice(["VIIRS_SNPP", "VIIRS_NOAA20"]),
            "confidence": random.choice([35.0, 65.0, 88.0, 92.0]),
            "frp": round(random.uniform(6.5, 68.0), 1),
            "daynight": 1
        })
    return records

def analyze_citizen_photo(image_path: str):
    img = Image.open(image_path)
    img_hash = str(imagehash.phash(img))
    
    clean_img = Image.new(img.mode, img.size)
    clean_img.putdata(list(img.getdata()))
    clean_img.save(image_path)

    results = yolo_model(image_path, verbose=False)
    label_ai = "bukan"
    conf_ai = 0.0

    if IS_CUSTOM_YOLO:
        for r in results:
            if len(r.boxes) > 0:
                best_box = max(r.boxes, key=lambda b: float(b.conf[0]))
                cls_id = int(best_box.cls[0])
                cls_name = yolo_model.names[cls_id].lower()
                conf_ai = round(float(best_box.conf[0]), 2)
                if "fire" in cls_name or "api" in cls_name:
                    label_ai = "api"
                elif "smoke" in cls_name or "asap" in cls_name:
                    label_ai = "asap"
    else:
        rgb_arr = np.array(img.convert("RGB").resize((160, 160)), dtype=float)
        r_ch, g_ch, b_ch = rgb_arr[:,:,0], rgb_arr[:,:,1], rgb_arr[:,:,2]
        fire_pixels = np.mean((r_ch > 180) & (g_ch > 70) & (g_ch < 190) & (b_ch < 90))
        smoke_pixels = np.mean((np.abs(r_ch - g_ch) < 18) & (np.abs(g_ch - b_ch) < 18) & (r_ch > 110) & (r_ch < 220))
        
        if fire_pixels > 0.06:
            label_ai = "api"
            conf_ai = round(min(0.96, 0.72 + fire_pixels * 1.5), 2)
        elif smoke_pixels > 0.22:
            label_ai = "asap"
            conf_ai = round(min(0.91, 0.68 + smoke_pixels * 0.6), 2)
        else:
            label_ai = "bukan"
            conf_ai = 0.85

    return img_hash, label_ai, conf_ai