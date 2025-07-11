# app.py

# --- Python Standard Library Imports ---
import datetime
import os
import base64
import time
import json
import logging
import sys
import threading
from typing import Dict, Any

# --- Third-party Library Imports ---
import cv2
import numpy as np
from rich import _console
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import BadRequest
from werkzeug.middleware.proxy_fix import ProxyFix

# --- Local Module Imports ---
import config
from utils import setup_logging, get_and_map_users_from_api
from services import ftp_service, rmq_service, db_service
from analysis.analysis_refactored import FaceAnalysisSystem
from services.static_service import simpan_dari_bytes

# --- Initial Setup ---
# setup_logging()
# logging.info("Flask application starting...")

# --- [PERBAIKAN] TAMBAHKAN BLOK INI UNTUK MEMASTIKAN LOG MUNCUL DI TERMINAL ---
logging.basicConfig(
    level=logging.DEBUG, # Ganti ke logging.DEBUG untuk detail lebih banyak
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Memaksa output ke terminal
)
# -----------------------------------------------------------------------------

# --- Initial Setup ---
# setup_logging() # <-- Beri komentar pada baris ini
logging.info("Flask application starting...") # <-- Log ini HARUS muncul sekarang

# --- Inisialisasi Aplikasi dan Model ---
# [PERBAIKAN] Inisialisasi face_analyzer dipindahkan ke atas agar bisa dicek lebih awal.
face_analyzer = FaceAnalysisSystem()

# [PERBAIKAN] Fail-Fast: Aplikasi akan keluar jika model inti tidak berhasil dimuat.
# Ini mencegah server berjalan dalam keadaan rusak (zombie state).
if not face_analyzer.is_ready():
    logging.critical("Sistem Analisis Wajah gagal diinisialisasi. Model penting tidak dimuat. Aplikasi akan berhenti.")
    sys.exit(1) # Keluar dari aplikasi dengan status error
else:
    logging.info("Sistem Analisis Wajah berhasil diinisialisasi dan siap menerima request.")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# --- State Management & Data Loading ---
user_details_map: Dict[str, Any] = {}
last_user_refresh_time: float = 0.0
training_lock = threading.Lock()
last_detection_timestamps: Dict[str, float] = {}

def refresh_users_if_needed():
    """
    [PERBAIKAN BARU] Fungsi untuk menyegarkan data pengguna dari API secara berkala.
    Ini memastikan data pengguna (seperti nama atau unit) selalu up-to-date.
    """
    global user_details_map, last_user_refresh_time
    current_time = time.time()
    # Segarkan data setiap 10 menit (600 detik) atau jika map masih kosong
    if not user_details_map or (current_time - last_user_refresh_time) > config.USER_REFRESH_INTERVAL_SECONDS:
        logging.info("Menyegarkan data pengguna dari API...")
        try:
            user_details_map = get_and_map_users_from_api()
            last_user_refresh_time = current_time
            logging.info(f"Berhasil menyegarkan data untuk {len(user_details_map)} pengguna.")
        except Exception as e:
            logging.error(f"Gagal menyegarkan data pengguna dari API: {e}", exc_info=True)

# ------------------------------------------------------------------
def _process_detection(person_data: Dict[str, Any], image: np.ndarray, lat: float, lon: float) -> bool:
    """
    1. Simpan foto RAW   (tanpa bounding‑box) → static + (opsional) FTP
    2. Buat foto Annot   (dengan bounding‑box) → static + FTP
    3. Tulis Annot URL ke Mongo history_ai
    4. Kirim metadata RAW ke queue RMQ hanya kalau absensi BERHASIL
    """
    # ----------- Persiapan dasar ------------
    guid = person_data["guid"]
    name = (person_data.get("nama")
            or person_data.get("name")
            or user_details_map.get(guid, {}).get("nama", "Nama Tidak Ditemukan"))

    now_epoch = int(time.time())
    last_detection_timestamps[guid] = now_epoch

    # ----------- 1. RAW  -------------------
    raw_fname  = f"raw_{guid}_{now_epoch}.jpg"
    _, raw_buf = cv2.imencode(".jpg", image)
    raw_bytes  = raw_buf.tobytes()
    raw_url    = simpan_dari_bytes(raw_bytes, guid, filename=raw_fname)       # local URL

    # upload RAW ke FTP (FE mungkin mau konsumsi langsung)
    try:
        if ftp_service.upload_to_ftp(raw_bytes, raw_fname):
            raw_url = f"{raw_fname}"
            logging.info("RAW upload FTP ✔")
    except Exception as e:
        logging.warning(f"RAW FTP error: {e}")

    # ----------- 2. Annotated --------------
    annotated = image.copy()
    box = person_data.get("box")
    if box:
        (x1, y1, w, h) = box
        x2, y2         = x1 + w, y1 + h
        color_map      = {'Marah': (0,0,255), 'Sedih': (255,0,0),
                          'Senang': (0,255,0), 'Netral': (255,255,0)}
        color          = color_map.get(person_data.get("mood", "Netral"), (0,255,0))
        label          = f"{name}|{person_data.get('mood')}|{person_data.get('keletihan',0):.1f}%"
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        cv2.rectangle(annotated, (x1, y2-35), (x2, y2), color, cv2.FILLED)
        cv2.putText(annotated, label, (x1+6, y2-6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)

    annot_fname  = f"annot_{guid}_{now_epoch}.jpg"
    _, annot_buf = cv2.imencode(".jpg", annotated)
    annot_bytes  = annot_buf.tobytes()
    annot_url    = simpan_dari_bytes(annot_bytes, guid, filename=annot_fname) # local

    # upload annotated ke FTP; FE bisa ‘view’ gambar bercorak kotak
    try:
        if ftp_service.upload_to_ftp(annot_bytes, annot_fname):
            annot_url = f"{annot_fname}"
            logging.info("Annotated upload FTP ✔")
    except Exception as e:
        logging.warning(f"Annotated FTP error: {e}")

    # ----------- 3. Simpan ke Mongo --------
    pesan_db = db_service.save_detection_history(person_data, annot_url)  # gunakan URL (lokal/FTP) annotated
    logging.info(f"[Mongo] {pesan_db}")

    # ----------- 4. Kirim atau tidak ke RMQ ?
    lower = pesan_db.lower()
    if ("sudah absen" in lower or "belum waktunya" in lower or "gagal" in lower):
        person_data["status_kirim"]  = "tidak_dikirim"
        person_data["presence_sent"] = False
        return False

    # Absensi BERHASIL ⇒ kirim metadata RAW
    rmq_service.publish_photo_metadata(raw_fname)
    person_data["status_kirim"]  = "berhasil"
    person_data["presence_sent"] = True
    return True
    # ------------------------------------------------------------------


# --- Flask Routes ---
@app.route('/')
def index():
    """Menampilkan halaman utama dengan daftar pengguna."""
    refresh_users_if_needed() # [PERBAIKAN] Panggil fungsi refresh
    return render_template('index.html', users=list(user_details_map.values()))

@app.route('/capture', methods=['POST'])
def capture():
    """Menangani penyimpanan gambar baru untuk training."""
    try:
        data = request.get_json()
        if not data or not all(k in data for k in ['image', 'name', 'guid']):
            raise BadRequest("Data tidak lengkap: 'image', 'name', 'guid' diperlukan.")

        user_guid = data['guid']
        user_name = data.get('nama') or data.get('name')

        # [PERBAIKAN] Penanganan error jika format base64 tidak valid
        try:
            image_data_b64 = data['image'].split(',')[1]
            img_bytes = base64.b64decode(image_data_b64)
        except (IndexError, base64.binascii.Error) as e:
            raise BadRequest(f"Format data gambar base64 tidak valid: {e}")

        user_folder = os.path.join(config.DATASET_PATH, f"{user_name.replace(' ', '_')}_{user_guid}")
        os.makedirs(user_folder, exist_ok=True)

        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'status': 'error', 'message': 'Data gambar tidak valid atau korup.'}), 400

        image_path = os.path.join(user_folder, f"capture_{int(time.time())}.jpg")
        cv2.imwrite(image_path, img)
        logging.info(f"Berhasil menyimpan gambar baru ke {image_path}")
        

        return jsonify({'status': 'success', 'message': f'Gambar untuk {user_name} telah disimpan!'})
    except BadRequest as e:
        logging.error(f"Request buruk di /capture: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logging.error(f"Error saat memproses gambar tangkapan: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Terjadi kesalahan internal saat menyimpan gambar.'}), 500

@app.route('/train', methods=['GET'])
def train_model():
    """Memicu proses training model AI secara aman."""
    if not training_lock.acquire(blocking=False):
        return jsonify({'status': 'error', 'message': 'Proses training lain sedang berjalan. Coba lagi nanti.'}), 409
    
    logging.info("Proses training dipicu oleh pengguna...")
    try:
        success, message = face_analyzer.train_model()
        status_code = 200 if success else 400
        return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code
    except Exception as e:
        logging.critical(f"Terjadi error tak terduga saat training: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Terjadi kesalahan fatal saat training.'}), 500
    finally:
        training_lock.release()
        logging.info("Proses training selesai, lock dilepaskan.")

@app.route('/recognize_frame', methods=['POST'])
def recognize_frame():
    """Endpoint inti untuk analisis frame."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            raise BadRequest("Request JSON tidak valid atau key 'image' tidak ada.")

        image_data_b64 = data['image'].split(',')[1]
        lat = data.get('latitude', 0.0)
        lon = data.get('longitude', 0.0)
        
        img_bytes = base64.b64decode(image_data_b64)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            logging.warning("Menerima data gambar yang tidak bisa di-decode oleh OpenCV.")
            return jsonify([])
            
    # [PERBAIKAN] Penanganan error yang lebih spesifik dan jelas
    except BadRequest as e:
        logging.error(f"Request buruk di /recognize_frame: {e}")
        return jsonify({'error': str(e)}), 400
    except (IndexError, base64.binascii.Error):
        logging.error("Gagal memproses data gambar: format base64 tidak valid.")
        return jsonify({'error': 'Format base64 tidak valid.'}), 400
    except Exception as e:
        logging.error(f"Error tak terduga saat memproses request frame: {e}", exc_info=True)
        return jsonify({'error': 'Kesalahan internal server.'}), 500

    # [PERBAIKAN] Pastikan data pengguna selalu yang terbaru sebelum analisis.
    refresh_users_if_needed()

    # Panggil metode analisis
    results = face_analyzer.analyze_image(img, user_details_map)

    for person_data in results:
        logging.debug(f"Detected person_data: {person_data}")
        user_guid = person_data.get('guid')
        if user_guid:
            current_time = time.time()
            last_seen = last_detection_timestamps.get(user_guid, 0)
            cooldown_expired = (current_time - last_seen) > config.DETECTION_COOLDOWN_SECONDS

            if cooldown_expired:
                success = _process_detection(person_data, img, lat, lon)
                person_data['presence_sent'] = success  # ✅ tandai apakah berhasil dikirim ke backend
                last_detection_timestamps[user_guid] = current_time  # Simpan waktu deteksi terbaru
            else:
                person_data['presence_sent'] = False  # ✅ masih dalam masa cooldown
        else:
            person_data['presence_sent'] = False  # ✅ tidak dikenali, tidak bisa dikirim
    
    success_count = sum(1 for person in results if person.get('presence_sent'))
    already_present_count = sum(1 for person in results if person.get('status_absen_message', '').lower().startswith("sudah absen"))
    fail_count = sum(
    1 for person in results 
    if person.get('presence_sent') is False and person.get('status_kirim') not in ['sudah_absen', 'belum_waktunya'] 
    )

    message_parts = []
   
    if success_count:
        message_parts.append(f"{success_count} berhasil")
    if already_present_count:
        message_parts.append(f"{already_present_count} sudah absen")
    if fail_count:
        message_parts.append(f"{fail_count} gagal dikirim")

    # Tambahkan ini sebelum return jsonify
    logging.debug(f"message_parts: {message_parts}")
    logging.debug(f"results: {json.dumps(results, indent=2, ensure_ascii=False)}")


    final_message = ', '.join(message_parts) or "Tidak ada deteksi"

    return jsonify({
        "status": "success" if fail_count == 0 else "partial",
        "message": final_message,
        "results": results
    })



@app.route('/get_training_stats')
def get_training_stats():
    """Menyediakan statistik jumlah gambar di dataset."""
    try:
        if not os.path.exists(config.DATASET_PATH):
            return jsonify({'image_count': 0})
        
        image_count = sum(len(files) for _, _, files in os.walk(config.DATASET_PATH))
        return jsonify({'image_count': image_count})
    except Exception as e:
        logging.error(f"Error menghitung statistik training: {e}", exc_info=True)
        return jsonify({'error': 'Gagal mengambil statistik.', 'detail': str(e)}), 500
    

# --- Main Execution ---
if __name__ == '__main__':
    # [PERBAIKAN] Memanggil refresh pengguna pertama kali saat startup
    refresh_users_if_needed()
    logging.debug(f"user_details_map keys: {list(user_details_map.keys())}")

    app.run(host='0.0.0.0', port=config.APP_PORT, debug=False, threaded=True)

