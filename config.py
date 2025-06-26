import os

# --- Application Configuration ---
APP_PORT = 6734

# --- Paths & Model Files ---
# Base directory of the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.clf')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.pkl')
DLIB_SHAPE_PREDICTOR = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, 'emotion_model.h5')

# --- External Services API ---
API_URL = "https://presensi-api.lskk.co.id/api/v1/user/public?id-institution=CMb80a&isDeleted=false"
GUID_INSTITUTION = "CMb80a"

# --- FTP Server Configuration ---
FTP_HOST = "ftp5.pptik.id"
FTP_PORT = 2121
FTP_USER = "monitoring"
# IMPORTANT: Use environment variables for passwords in a real application.
# For example: os.environ.get('FTP_PASS', 'default_password')
FTP_PASS = "Tpm0ni23!n6"
FTP_FOLDER = "/presensi" # Note: The leading slash is handled in the FTP service.
FTP_BASE_URL = "https://monja-file.pptik.id/v1/view?path=" # The folder name is appended later

# --- RabbitMQ (RMQ) Configuration ---
# IMPORTANT: Use environment variables for passwords.
RMQ_PASSWORD = "PPTIK@|PASSWORD"
RMQ_URI = f"amqp://absensi:PPTIK@|PASSWORD@cloudabsensi.pptik.id:5672/%2fabsensi?heartbeat=600&blocked_connection_timeout=300"
RMQ_QUEUE = "presence-2024"

# --- Cooldown Configuration ---
# Cooldown in seconds between sending presence data for the same person.
DETECTION_COOLDOWN_SECONDS = 60

# --- Emotion Detection Labels ---
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

