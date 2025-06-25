# import os
# import requests
# import cv2
# import numpy as np
# from flask import Flask, render_template, request, jsonify, redirect, url_for
# from sklearn.neighbors import KNeighborsClassifier
# import joblib
# import pickle
# import base64
# import dlib
# from scipy.spatial import distance as dist
# from tensorflow.keras.models import load_model
# import face_recognition

# # --- Inisialisasi Aplikasi Flask ---
# app = Flask(__name__)

# # --- Konfigurasi Path ---
# DATASET_PATH = 'dataset'
# MODEL_PATH = 'trained_model.clf'
# LABELS_PATH = 'labels.pkl'
# DLIB_SHAPE_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'
# EMOTION_MODEL_PATH = 'emotion_model.h5'

# # URL API
# API_URL = "https://presensi-api.lskk.co.id/api/v1/user/public?id-institution=CMb80a&isDeleted=false"

# # --- MEMUAT SEMUA MODEL DAN DATA USER SAAT STARTUP ---
# knn_clf = None
# detector = None
# predictor = None
# emotion_classifier = None
# user_details_map = {} # Dictionary untuk menyimpan detail user

# def get_and_map_users_from_api():
#     global user_details_map
#     print("Fetching and mapping user details from API...")
#     try:
#         response = requests.get(API_URL)
#         response.raise_for_status()
#         users = response.json().get('data', [])
#         for user in users:
#             label_key = f"{user['name'].replace(' ', '_')}_{user['guid']}"
#             user_details_map[label_key] = user
#         print(f"Successfully mapped {len(user_details_map)} users.")
#         return users
#     except requests.exceptions.RequestException as e:
#         print(f"API Error: {e}")
#         return []

# get_and_map_users_from_api()

# try:
#     if os.path.exists(MODEL_PATH):
#         print("Loading face recognition model...")
#         knn_clf = joblib.load(MODEL_PATH)
#         print("Face recognition model loaded.")
#     else:
#         print("Warning: Face recognition model not found. Please train the model first.")
# except (EOFError, pickle.UnpicklingError):
#     print("ERROR: Model file 'trained_model.clf' is corrupt. Please delete it and retrain.")

# if os.path.exists(DLIB_SHAPE_PREDICTOR):
#     print("Loading Dlib model...")
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR)
#     print("Dlib model loaded.")
# else:
#     print(f"Error: Dlib model '{DLIB_SHAPE_PREDICTOR}' not found!")

# if os.path.exists(EMOTION_MODEL_PATH):
#     print("Loading emotion detection model...")
#     emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)
#     EMOTION_LABELS = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']
#     print("Emotion detection model loaded.")
# else:
#     print(f"Error: Emotion model '{EMOTION_MODEL_PATH}' not found!")


# # --- Fungsi Helper untuk Analisis Gambar ---
# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# def analyze_image_for_faces(image):
#     if detector is None or predictor is None: return []
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     rects = detector(gray, 0)
#     results = []
    
#     for rect in rects:
#         (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
#         result = {"box": (x, y, w, h), "name": "Tidak Dikenal", "unit": "-"}

#         if knn_clf:
#             face_location = [(rect.top(), rect.right(), rect.bottom(), rect.left())]
#             encodings = face_recognition.face_encodings(rgb_image, face_location)
#             if encodings:
#                 closest_distances = knn_clf.kneighbors(encodings, n_neighbors=1)
#                 if closest_distances[0][0][0] <= 0.5:
#                     label_key = knn_clf.predict(encodings)[0]
#                     user_info = user_details_map.get(label_key)
#                     if user_info:
#                         result["name"] = user_info.get("name", "N/A")
#                         result["unit"] = user_info.get("unit", "N/A")
        
#         shape = predictor(gray, rect)
#         shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        
#         leftEye = shape_np[42:48]; rightEye = shape_np[36:42]
#         ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
#         result["fatigue"] = "Mengantuk" if ear < 0.25 else "Sadar"
        
#         if emotion_classifier:
#             roi_gray = cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)
#             roi = roi_gray.astype('float') / 255.0; roi = np.asarray(roi)
#             roi = np.expand_dims(roi, axis=0); roi = np.expand_dims(roi, axis=-1)
#             preds = emotion_classifier.predict(roi, verbose=0)[0]
#             result["emotion"] = EMOTION_LABELS[preds.argmax()]
        
#         results.append(result)
#     return results

# # --- Routes Aplikasi ---
# @app.route('/')
# def index():
#     users = list(user_details_map.values()) if user_details_map else get_and_map_users_from_api()
#     return render_template('index.html', users=users)

# @app.route('/capture', methods=['POST'])
# def capture():
#     data = request.get_json()
#     user_guid = data['guid']; user_name = data['name']
#     image_data = data['image'].split(',')[1]
#     user_folder = os.path.join(DATASET_PATH, f"{user_name.replace(' ', '_')}_{user_guid}")
#     if not os.path.exists(user_folder): os.makedirs(user_folder)
#     image_path = os.path.join(user_folder, f"image_{len(os.listdir(user_folder)) + 1}.jpg")
#     try:
#         decoded_data = base64.b64decode(image_data)
#         nparr = np.frombuffer(decoded_data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         cv2.imwrite(image_path, img)
#         return jsonify({'status': 'success', 'message': f'Gambar untuk {user_name} disimpan!'})
#     except Exception as e:
#         print(f"Error processing image: {e}")
#         return jsonify({'status': 'error', 'message': 'Gagal menyimpan gambar.'}), 500

# @app.route('/train', methods=['GET'])
# def train_model():
#     face_encodings, labels = [], []
#     print("Starting model training...")
#     for user_folder in os.listdir(DATASET_PATH):
#         user_path = os.path.join(DATASET_PATH, user_folder)
#         if not os.path.isdir(user_path): continue
#         for image_name in os.listdir(user_path):
#             image_path = os.path.join(user_path, image_name)
#             if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
#             image = face_recognition.load_image_file(image_path)
#             encodings = face_recognition.face_encodings(image)
#             if encodings:
#                 face_encodings.append(encodings[0]); labels.append(user_folder)
    
#     if not face_encodings:
#         print("Training failed: No faces found in the dataset to train.")
#         return jsonify({'status': 'error', 'message': 'Gagal melatih: Tidak ada wajah di dataset.'}), 400

#     n_neighbors = int(round(np.sqrt(len(face_encodings))))
#     knn_clf_new = KNeighborsClassifier(n_neighbors=max(1, n_neighbors), algorithm='ball_tree', weights='distance')
#     knn_clf_new.fit(face_encodings, labels)
    
#     joblib.dump(knn_clf_new, MODEL_PATH)
#     with open(LABELS_PATH, 'wb') as f: pickle.dump(labels, f)

#     print("Training completed successfully! Reloading model in memory...")
#     globals()['knn_clf'] = knn_clf_new
    
#     return jsonify({'status': 'success', 'message': 'Model berhasil dilatih ulang!'})

# @app.route('/recognize_frame', methods=['POST'])
# def recognize_frame():
#     if 'image' not in request.json: return jsonify({"error": "No image data"}), 400
#     image_data = request.json['image'].split(',')[1]
    
#     try:
#         decoded_data = base64.b64decode(image_data)
#         nparr = np.frombuffer(decoded_data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if img is None: return jsonify({"error": "Invalid image data"}), 400
#     except Exception:
#         return jsonify({"error": "Invalid base64 string"}), 400

#     results = analyze_image_for_faces(img)
#     return jsonify(results)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=6734, debug=False)


# import os
# import requests
# import cv2
# import numpy as np
# from flask import Flask, render_template, request, jsonify, redirect, url_for
# from sklearn.neighbors import KNeighborsClassifier
# import joblib
# import pickle
# import base64
# import dlib
# from scipy.spatial import distance as dist
# from tensorflow.keras.models import load_model
# import face_recognition
# import datetime
# import time
# import json
# import pika
# from ftplib import FTP_TLS
# from io import BytesIO

# # --- Inisialisasi Aplikasi Flask ---
# app = Flask(__name__)

# # --- Konfigurasi Path & API ---
# DATASET_PATH = 'dataset'
# MODEL_PATH = 'trained_model.clf'
# LABELS_PATH = 'labels.pkl'
# DLIB_SHAPE_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'
# EMOTION_MODEL_PATH = 'emotion_model.h5'
# API_URL = "https://presensi-api.lskk.co.id/api/v1/user/public?id-institution=CMb80a&isDeleted=false"
# GUID_INSTITUTION = "CMb80a"

# # --- KONFIGURASI BARU UNTUK FTP & RMQ ---
# FTP_HOST = "ftp5.pptik.id"
# FTP_PORT = 2121
# FTP_USER = "monitoring"
# FTP_PASS = "Tpm0ni23!n6"
# FTP_FOLDER = "/monitoring"
# FTP_BASE_URL = "https://monja-file.pptik.id/v1/view?path=monitoring"

# # GANTI 'PASSWORD_ANDA' DENGAN PASSWORD RMQ YANG BENAR
# RMQ_PASSWORD = 'PPTIK@|PASSWORD' 
# RMQ_URI = f"amqp://absensi:{RMQ_PASSWORD}@cloudabsensi.pptik.id:5672/%2fabsensi?heartbeat=600&blocked_connection_timeout=300"
# RMQ_QUEUE = "presence-2024"

# # --- Manajemen State untuk Cooldown ---
# last_detection_timestamps = {}
# DETECTION_COOLDOWN_SECONDS = 60

# # --- MEMUAT MODEL DAN DATA USER ---
# knn_clf = None; detector = None; predictor = None; emotion_classifier = None; user_details_map = {}

# def get_and_map_users_from_api():
#     global user_details_map
#     print("Fetching and mapping user details from API...")
#     try:
#         response = requests.get(API_URL)
#         response.raise_for_status()
#         users = response.json().get('data', [])
#         for user in users:
#             label_key = f"{user['name'].replace(' ', '_')}_{user['guid']}"
#             user_details_map[label_key] = user
#         print(f"Successfully mapped {len(user_details_map)} users.")
#         return users
#     except requests.exceptions.RequestException as e:
#         print(f"API Error: {e}")
#         return []

# get_and_map_users_from_api()

# try:
#     if os.path.exists(MODEL_PATH):
#         print("Loading face recognition model...")
#         knn_clf = joblib.load(MODEL_PATH)
#         print("Face recognition model loaded.")
# except (EOFError, pickle.UnpicklingError):
#     print("ERROR: Model file 'trained_model.clf' is corrupt. Please delete it and retrain.")

# if os.path.exists(DLIB_SHAPE_PREDICTOR):
#     print("Loading Dlib model...")
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR)
#     print("Dlib model loaded.")
# else:
#     print(f"Error: Dlib model '{DLIB_SHAPE_PREDICTOR}' not found!")

# if os.path.exists(EMOTION_MODEL_PATH):
#     print("Loading emotion detection model...")
#     emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)
#     EMOTION_LABELS = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']
#     print("Emotion detection model loaded.")
# else:
#     print(f"Error: Emotion model '{EMOTION_MODEL_PATH}' not found!")


# # --- FUNGSI HELPER ---
# def upload_to_ftp(image_bytes, remote_filename):
#     try:
#         with FTP_TLS() as ftp:
#             ftp.connect(FTP_HOST, FTP_PORT)
#             ftp.login(FTP_USER, FTP_PASS)
#             ftp.prot_p()
#             if FTP_FOLDER not in ftp.nlst(): ftp.mkd(FTP_FOLDER)
#             ftp.cwd(FTP_FOLDER)
#             with BytesIO(image_bytes) as file_obj:
#                 ftp.storbinary(f'STOR {remote_filename}', file_obj)
#             print(f"FTP Upload Success: {remote_filename}")
#             return True, None
#     except Exception as e:
#         print(f"FTP Upload Error: {e}")
#         return False, str(e)

# def publish_to_rmq(payload):
#     try:
#         params = pika.URLParameters(RMQ_URI)
#         connection = pika.BlockingConnection(params)
#         channel = connection.channel()
#         channel.queue_declare(queue=RMQ_QUEUE, durable=True)
#         message_body = json.dumps(payload)
#         channel.basic_publish(exchange='', routing_key=RMQ_QUEUE, body=message_body, properties=pika.BasicProperties(delivery_mode=2))
#         print(f"RMQ Message Published to queue '{RMQ_QUEUE}'")
#         connection.close()
#         return True, None
#     except Exception as e:
#         print(f"RMQ Publish Error: {e}")
#         return False, str(e)

# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4]); C = dist.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# def analyze_image_for_faces(image):
#     if detector is None or predictor is None: return []
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     rects = detector(gray, 0)
#     results = []
#     for rect in rects:
#         (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
#         result = {"box": (x, y, w, h), "name": "Tidak Dikenal", "unit": "-", "guid": None}
#         if knn_clf:
#             encodings = face_recognition.face_encodings(rgb_image, [(rect.top(), rect.right(), rect.bottom(), rect.left())])
#             if encodings:
#                 closest_distances = knn_clf.kneighbors(encodings, n_neighbors=1)
#                 if closest_distances[0][0][0] <= 0.5:
#                     label_key = knn_clf.predict(encodings)[0]
#                     user_info = user_details_map.get(label_key)
#                     if user_info:
#                         result.update(user_info) # Memasukkan semua info user
#         shape = predictor(gray, rect)
#         shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
#         result["fatigue"] = "Mengantuk" if (eye_aspect_ratio(shape_np[42:48]) + eye_aspect_ratio(shape_np[36:42])) / 2.0 < 0.25 else "Sadar"
#         if emotion_classifier:
#             roi = gray[y:y + h, x:x + w]; roi = cv2.resize(roi, (48, 48))
#             roi = roi.astype('float') / 255.0; roi = np.asarray(roi); roi = np.expand_dims(roi, axis=0); roi = np.expand_dims(roi, axis=-1)
#             result["emotion"] = EMOTION_LABELS[emotion_classifier.predict(roi, verbose=0)[0].argmax()]
#         results.append(result)
#     return results

# # --- Routes Aplikasi ---
# @app.route('/')
# def index():
#     users = list(user_details_map.values()) if user_details_map else get_and_map_users_from_api()
#     return render_template('index.html', users=users)

# @app.route('/capture', methods=['POST'])
# def capture():
#     data = request.get_json()
#     user_guid, user_name = data['guid'], data['name']
#     image_data = data['image'].split(',')[1]
#     user_folder = os.path.join(DATASET_PATH, f"{user_name.replace(' ', '_')}_{user_guid}")
#     if not os.path.exists(user_folder): os.makedirs(user_folder)
#     image_path = os.path.join(user_folder, f"image_{len(os.listdir(user_folder)) + 1}.jpg")
#     try:
#         decoded_data = base64.b64decode(image_data)
#         img = cv2.imdecode(np.frombuffer(decoded_data, np.uint8), cv2.IMREAD_COLOR)
#         cv2.imwrite(image_path, img)
#         return jsonify({'status': 'success', 'message': f'Gambar untuk {user_name} disimpan!'})
#     except Exception as e:
#         print(f"Error processing image: {e}")
#         return jsonify({'status': 'error', 'message': 'Gagal menyimpan gambar.'}), 500

# @app.route('/train', methods=['GET'])
# def train_model():
#     face_encodings, labels = [], []
#     print("Starting model training...")
#     for user_folder in os.listdir(DATASET_PATH):
#         user_path = os.path.join(DATASET_PATH, user_folder)
#         if not os.path.isdir(user_path): continue
#         for image_name in os.listdir(user_path):
#             image_path = os.path.join(user_path, image_name)
#             if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
#             image = face_recognition.load_image_file(image_path)
#             encodings = face_recognition.face_encodings(image)
#             if encodings:
#                 face_encodings.append(encodings[0]); labels.append(user_folder)
    
#     if not face_encodings:
#         return jsonify({'status': 'error', 'message': 'Gagal melatih: Tidak ada wajah di dataset.'}), 400

#     n_neighbors = int(round(np.sqrt(len(face_encodings))))
#     knn_clf_new = KNeighborsClassifier(n_neighbors=max(1, n_neighbors), algorithm='ball_tree', weights='distance')
#     knn_clf_new.fit(face_encodings, labels)
    
#     joblib.dump(knn_clf_new, MODEL_PATH)
#     globals()['knn_clf'] = knn_clf_new
#     return jsonify({'status': 'success', 'message': 'Model berhasil dilatih ulang!'})

# @app.route('/recognize_frame', methods=['POST'])
# def recognize_frame():
#     image_data = request.json['image'].split(',')[1]
#     decoded_data = base64.b64decode(image_data)
#     img = cv2.imdecode(np.frombuffer(decoded_data, np.uint8), cv2.IMREAD_COLOR)
#     if img is None: return jsonify([])

#     results = analyze_image_for_faces(img)
    
#     if results and results[0].get('guid'):
#         person = results[0]
#         user_guid = person['guid']
#         current_time = time.time()
#         last_sent_time = last_detection_timestamps.get(user_guid, 0)
        
#         if (current_time - last_sent_time) > DETECTION_COOLDOWN_SECONDS:
#             print(f"Cooldown finished for user {user_guid}. Processing presence...")
#             _, buffer = cv2.imencode('.jpg', img)
#             remote_filename = f"detection_{user_guid}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#             ftp_ok, _ = upload_to_ftp(buffer.tobytes(), remote_filename)
            
#             if ftp_ok:
#                 payload = {
#                     "pattern": "daily-report",
#                     "data": {
#                         "id": user_guid, "type": "AI_Detection", "presenceType": "Daily Presence",
#                         "description": f"Deteksi wajah berhasil: {person.get('name', 'N/A')}",
#                         "imageUrl": f"{FTP_BASE_URL}/{remote_filename}",
#                         "latitude": -6.9293731, "longitude": 107.6268769,
#                         "guidInstitution": GUID_INSTITUTION,
#                         "hour": datetime.datetime.now().strftime("%H.%M")
#                     }
#                 }
#                 publish_to_rmq(payload)
#                 last_detection_timestamps[user_guid] = current_time
#                 results[0]['presence_sent'] = True
                
#     return jsonify(results)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=6734, debug=False)

import os
import requests
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pickle
import base64
import dlib
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
import face_recognition
import datetime
import time
import json
import pika
from ftplib import FTP_TLS
from io import BytesIO
import logging # <-- Impor baru untuk logging

# --- KONFIGURASI LOGGING ---
# Mengatur format log agar menampilkan waktu, level, dan pesan
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
logging.info("Flask application starting...")

# --- Konfigurasi Path & API ---
DATASET_PATH = 'dataset'
MODEL_PATH = 'trained_model.clf'
LABELS_PATH = 'labels.pkl'
DLIB_SHAPE_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'
EMOTION_MODEL_PATH = 'emotion_model.h5'
API_URL = "https://presensi-api.lskk.co.id/api/v1/user/public?id-institution=CMb80a&isDeleted=false"
GUID_INSTITUTION = "CMb80a"

# --- Konfigurasi FTP & RMQ ---
FTP_HOST = "ftp5.pptik.id"
FTP_PORT = 2121
FTP_USER = "monitoring"
FTP_PASS = "Tpm0ni23!n6"
FTP_FOLDER = "/monitoring"
FTP_BASE_URL = "https://monja-file.pptik.id/v1/view?path=monitoring"

RMQ_PASSWORD = 'PPTIK@|PASSWORD' # <-- PASTIKAN INI DIGANTI
RMQ_URI = f"amqp://absensi:{RMQ_PASSWORD}@cloudabsensi.pptik.id:5672/%2fabsensi?heartbeat=600&blocked_connection_timeout=300"
RMQ_QUEUE = "presence-2024"

# --- Manajemen State untuk Cooldown ---
last_detection_timestamps = {}
DETECTION_COOLDOWN_SECONDS = 60

# --- MEMUAT MODEL DAN DATA USER ---
knn_clf = None; detector = None; predictor = None; emotion_classifier = None; user_details_map = {}

def get_and_map_users_from_api():
    global user_details_map
    logging.info("Fetching and mapping user details from API...")
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        users = response.json().get('data', [])
        for user in users:
            label_key = f"{user['name'].replace(' ', '_')}_{user['guid']}"
            user_details_map[label_key] = user
        logging.info(f"Successfully mapped {len(user_details_map)} users.")
        return users
    except requests.exceptions.RequestException as e:
        logging.error(f"API Error: {e}")
        return []

get_and_map_users_from_api()

try:
    if os.path.exists(MODEL_PATH):
        logging.info(f"Loading face recognition model from {MODEL_PATH}...")
        knn_clf = joblib.load(MODEL_PATH)
        logging.info("Face recognition model loaded successfully.")
    else:
        logging.warning("Face recognition model not found. Please train the model first.")
except (EOFError, pickle.UnpicklingError):
    logging.error(f"Model file '{MODEL_PATH}' is corrupt. Please delete it and retrain.")

if os.path.exists(DLIB_SHAPE_PREDICTOR):
    logging.info(f"Loading Dlib model from {DLIB_SHAPE_PREDICTOR}...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR)
    logging.info("Dlib model loaded successfully.")
else:
    logging.critical(f"Dlib model '{DLIB_SHAPE_PREDICTOR}' not found! Face detection will not work.")

if os.path.exists(EMOTION_MODEL_PATH):
    logging.info(f"Loading emotion detection model from {EMOTION_MODEL_PATH}...")
    emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)
    EMOTION_LABELS = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']
    logging.info("Emotion detection model loaded successfully.")
else:
    logging.critical(f"Emotion model '{EMOTION_MODEL_PATH}' not found! Emotion detection will not work.")


# --- FUNGSI HELPER ---
def upload_to_ftp(image_bytes, remote_filename):
    logging.info(f"Attempting to upload '{remote_filename}' to FTP...")
    try:
        with FTP_TLS() as ftp:
            ftp.connect(FTP_HOST, FTP_PORT, timeout=10)
            ftp.login(FTP_USER, FTP_PASS)
            ftp.prot_p()
            if FTP_FOLDER not in ftp.nlst(): ftp.mkd(FTP_FOLDER)
            ftp.cwd(FTP_FOLDER)
            with BytesIO(image_bytes) as file_obj:
                ftp.storbinary(f'STOR {remote_filename}', file_obj)
            logging.info(f"FTP Upload Success: '{remote_filename}'")
            return True
    except Exception as e:
        logging.error(f"FTP Upload Error: {e}")
        return False

def publish_to_rmq(payload):
    logging.info(f"Attempting to publish message to RMQ queue '{RMQ_QUEUE}'...")
    try:
        params = pika.URLParameters(RMQ_URI)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue=RMQ_QUEUE, durable=True)
        message_body = json.dumps(payload, indent=2)
        channel.basic_publish(exchange='', routing_key=RMQ_QUEUE, body=message_body, properties=pika.BasicProperties(delivery_mode=2))
        logging.info("RMQ Message Published successfully.")
        connection.close()
        return True
    except Exception as e:
        logging.error(f"RMQ Publish Error: {e}")
        return False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4]); C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def analyze_image_for_faces(image):
    if detector is None: return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = []
    for rect in detector(gray, 0):
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        result = {"box": (x, y, w, h), "name": "Tidak Dikenal", "unit": "-", "guid": None}
        if knn_clf:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image, [(y, x + w, y + h, x)])
            if encodings:
                closest_distances = knn_clf.kneighbors(encodings, n_neighbors=1)
                if closest_distances[0][0][0] <= 0.5:
                    user_info = user_details_map.get(knn_clf.predict(encodings)[0])
                    if user_info: result.update(user_info)
        if predictor:
            shape = predictor(gray, rect)
            shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            result["fatigue"] = "Mengantuk" if (eye_aspect_ratio(shape_np[42:48]) + eye_aspect_ratio(shape_np[36:42])) / 2.0 < 0.25 else "Sadar"
        if emotion_classifier:
            roi = gray[y:y + h, x:x + w]; roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float') / 255.0; roi = np.asarray(roi); roi = np.expand_dims(roi, axis=0); roi = np.expand_dims(roi, axis=-1)
            result["emotion"] = EMOTION_LABELS[emotion_classifier.predict(roi, verbose=0)[0].argmax()]
        results.append(result)
    return results

# --- Routes Aplikasi ---
@app.route('/')
def index():
    logging.info(f"Request received for main page (User-Agent: {request.user_agent.string})")
    users = list(user_details_map.values())
    return render_template('index.html', users=users)

@app.route('/capture', methods=['POST'])
def capture():
    user_guid = request.json.get('guid', 'unknown')
    logging.info(f"Capture request received for user GUID: {user_guid}")
    user_name = request.json['name']
    image_data = request.json['image'].split(',')[1]
    user_folder = os.path.join(DATASET_PATH, f"{user_name.replace(' ', '_')}_{user_guid}")
    if not os.path.exists(user_folder): os.makedirs(user_folder); logging.info(f"Created new dataset folder: {user_folder}")
    image_path = os.path.join(user_folder, f"image_{len(os.listdir(user_folder)) + 1}.jpg")
    try:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(image_path, img)
        logging.info(f"Image saved successfully to {image_path}")
        return jsonify({'status': 'success', 'message': f'Gambar untuk {user_name} disimpan!'})
    except Exception as e:
        logging.error(f"Error processing captured image: {e}")
        return jsonify({'status': 'error', 'message': 'Gagal menyimpan gambar.'}), 500

@app.route('/train', methods=['GET'])
def train_model():
    logging.info("Training process initiated by user.")
    face_encodings, labels = [], []
    for user_folder in os.listdir(DATASET_PATH):
        user_path = os.path.join(DATASET_PATH, user_folder)
        if not os.path.isdir(user_path): continue
        for image_name in os.listdir(user_path):
            image_path = os.path.join(user_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                face_encodings.append(encodings[0]); labels.append(user_folder)
            else:
                logging.warning(f"No face found in image: {image_path}")
    
    if not face_encodings:
        logging.error("Training failed: No faces with encodings found in the dataset.")
        return jsonify({'status': 'error', 'message': 'Gagal melatih: Tidak ada wajah di dataset.'}), 400

    n_neighbors = int(round(np.sqrt(len(face_encodings))))
    logging.info(f"Training KNN model with {len(face_encodings)} samples and n_neighbors={n_neighbors}...")
    knn_clf_new = KNeighborsClassifier(n_neighbors=max(1, n_neighbors), algorithm='ball_tree', weights='distance')
    knn_clf_new.fit(face_encodings, labels)
    
    joblib.dump(knn_clf_new, MODEL_PATH)
    globals()['knn_clf'] = knn_clf_new
    logging.info("Training completed and model reloaded successfully!")
    return jsonify({'status': 'success', 'message': 'Model berhasil dilatih ulang!'})

@app.route('/recognize_frame', methods=['POST'])
def recognize_frame():
    try:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(request.json['image'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        if img is None: return jsonify([])
    except Exception:
        return jsonify([])

    results = analyze_image_for_faces(img)
    
    if results and results[0].get('guid'):
        person = results[0]
        user_guid = person['guid']
        current_time = time.time()
        
        if (current_time - last_detection_timestamps.get(user_guid, 0)) > DETECTION_COOLDOWN_SECONDS:
            logging.info(f"User '{person.get('name')}' detected. Cooldown passed. Processing presence...")
            last_detection_timestamps[user_guid] = current_time
            
            _, buffer = cv2.imencode('.jpg', img); image_bytes = buffer.tobytes()
            remote_filename = f"detection_{user_guid}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            
            if upload_to_ftp(image_bytes, remote_filename):
                payload = { "pattern": "daily-report", "data": {
                        "id": user_guid, "type": "AI_Detection", "presenceType": "Daily Presence",
                        "description": f"Deteksi wajah berhasil: {person.get('name', 'N/A')}",
                        "imageUrl": f"{FTP_BASE_URL}/{remote_filename}",
                        "latitude": -6.9293731, "longitude": 107.6268769,
                        "guidInstitution": GUID_INSTITUTION, "hour": datetime.datetime.now().strftime("%H.%M")
                    }}
                publish_to_rmq(payload)
                results[0]['presence_sent'] = True
        else:
            logging.info(f"User '{person.get('name')}' detected. Cooldown active. Skipping.")
            
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6734, debug=False)