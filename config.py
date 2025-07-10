# import os

# # --- Application Configuration ---
# APP_PORT = 6734

# # --- Paths & Model Files ---
# # Base directory of the application
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
# MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.clf')
# LABELS_PATH = os.path.join(BASE_DIR, 'labels.pkl')
# DLIB_SHAPE_PREDICTOR = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
# EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'emotion_model.h5')
# STATIC_USERS_PATH = os.path.join(BASE_DIR, 'static_users.json')

# # --- External Services API ---
# API_URL = "https://presensi-api.lskk.co.id/api/v1/user/public?id-institution=CMb80a&isDeleted=false"
# GUID_INSTITUTION = "CMb80a"

# #DATABASE CONFIGURATION
# MONGO_URI = os.environ.get("MONGO_URI", "mongodb://naungilmu:n4un97!2@database2.pptik.id/naungilmu") 
# MONGO_DB_NAME = "naungilmu"

# # --- FTP Server Configuration ---
# FTP_HOST = "ftp5.pptik.id"
# FTP_PORT = 2121
# FTP_USER = "monitoring"
# # IMPORTANT: Use environment variables for passwords in a real application.
# # For example: os.environ.get('FTP_PASS', 'default_password')
# FTP_PASS = "Tpm0ni23!n6"
# FTP_FOLDER = "/presensi" # Note: The leading slash is handled in the FTP service.
# FTP_BASE_URL = "https://monja-file.pptik.id/v1/view?path=" # The folder name is appended later

# # --- RabbitMQ (RMQ) Configuration ---
# # IMPORTANT: Use environment variables for passwords.
# RMQ_PASSWORD = "PPTIK@|PASSWORD"
# RMQ_URI = f"amqp://absensi:PPTIK@|PASSWORD@cloudabsensi.pptik.id:5672/%2fabsensi?heartbeat=600&blocked_connection_timeout=300"
# RMQ_QUEUE = "presence-2024"

# # --- Cooldown Configuration ---
# # Cooldown in seconds between sending presence data for the same person.
# DETECTION_COOLDOWN_SECONDS = 60

# # [PERBAIKAN] RMQ #2 - For AI Presence Notification - Kredensial & Konfigurasi diperbarui
# RMQ2_USER = "ai-presensi"
# RMQ2_PASS = "BtRD6f9oG29c"
# RMQ2_HOST = "rmq230.pptik.id"
# RMQ2_PORT = 5672
# RMQ2_VHOST = "/ai-presensi"
# RMQ2_URI = f"amqp://ai-presensi:BtRD6f9oG29c@rmq230.pptik.id:5672/%2fai-presensi?heartbeat=600&blocked_connection_timeout=300"
# RMQ2_QUEUE = "presensi"

# # Hardcoded Camera GUID for RMQ2
# CAMERA_GUID = "CAM-P0721-DEVICE-USER"

# # --- Cooldown Configuration ---
# DETECTION_COOLDOWN_SECONDS = 60

# # --- Emotion Detection Labels ---
# EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



import os

# --- Paths & Model Files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
STATIC_USERS_PATH = os.path.join(BASE_DIR, 'static_users.json')
# Model untuk face recognition
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.clf') 
# Model untuk fitur wajah (mata, hidung, dll.)
# DLIB_SHAPE_PREDICTOR = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
DLIB_SHAPE_PREDICTOR = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
MODEL_PATH = os.path.join(BASE_DIR, 'knn_model.joblib')
# Model untuk deteksi emosi
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'emotion_model.h5')

# --- Application & API Configuration ---
# APP_PORT = 6734
# API_URL = "https://presensi-api.lskk.co.id/api/v1/user/public?id-institution=CMb80a&isDeleted=false"
GUID_INSTITUTION = "CMb80a"
CAMERA_GUID = "CAM-P0721" # GUID Kamera untuk notifikasi

APP_PORT = 6734
API_URL = "http://localhost:4000/karyawan/get"

# --- Database & Services Configuration ---
MONGO_URI = "mongodb+srv://wh891706:Jz1X5pslUXxCBw9s@cluster0.vt4xkx8.mongodb.net"
MONGO_DB_NAME = "presensi_ai"

FTP_HOST = "ftp5.pptik.id"
FTP_PORT = 2121
FTP_USER = "monitoring"
FTP_PASS = "Tpm0ni23!n6"
FTP_FOLDER = "/presensi"
FTP_BASE_URL = "https://monja-file.pptik.id/v1/view?path="

# RMQ #1 - Untuk data presensi utama
RMQ_URI = "amqp://absensi:PPTIK@|PASSWORD@cloudabsensi.pptik.id:5672/%2fabsensi?heartbeat=600&blocked_connection_timeout=300"
RMQ_QUEUE = "presence-2024"

# RMQ #2 - Untuk notifikasi AI Presensi
RMQ2_URI = "amqp://ai-presensi:BtRD6f9oG29c@rmq230.pptik.id:5672/%2fai-presensi?heartbeat=600&blocked_connection_timeout=300"
RMQ2_QUEUE = "presensi"


# --- Analysis Configuration ---
FATIGUE_EAR_THRESHOLD = 0.25 
EAR_THRESHOLD_CLOSED = 0.20  # Tambahkan baris ini
EAR_THRESHOLD_OPEN = 0.30    # Tambahkan baris ini
DETECTION_COOLDOWN_SECONDS = 60
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# EMOTION_LABELS_ENGLISH = ['angry', 'happy', 'neutral', 'sad']
EMOTION_MAP_TO_INDONESIAN = {
    'angry': 'Marah',
    'happy': 'Senang',
    'neutral': 'Netral',
    'sad': 'Sedih'
}

KNN_DISTANCE_THRESHOLD = 0.4

# Ukuran target (lebar, tinggi) untuk input model deteksi emosi.
EMOTION_TARGET_SIZE = (48, 48)

# Indeks landmark dlib untuk mata kiri dan kanan
LEFT_EYE_INDICES = list(range(42, 48))
RIGHT_EYE_INDICES = list(range(36, 42))

# Label emosi (pastikan urutannya sesuai dengan output model Anda)
EMOTION_LABELS_ENGLISH = ["angry", "happy", "neutral", "sad"] # Sesuaikan urutan ini!
EMOTION_LABELS_INDONESIAN = ["Marah", "Senang", "Netral", "Sedih"]

# Mapping dari label Inggris ke Indonesia untuk hasil akhir
EMOTION_MAP_TO_INDONESIAN = {
    "happy": "Senang",
    "angry": "Marah",
    "sad": "Sedih",
    "neutral": "Netral"
}

# Ambang batas Eye Aspect Ratio (EAR) untuk deteksi keletihan
# Sesuaikan nilai ini berdasarkan pengujian
EAR_THRESHOLD_OPEN = 0.25   # EAR saat mata terbuka normal
EAR_THRESHOLD_CLOSED = 0.15 # EAR saat mata dianggap tertutup
USER_REFRESH_INTERVAL_SECONDS= 180  # Interval untuk memperbarui data pengguna dari API (1 jam)