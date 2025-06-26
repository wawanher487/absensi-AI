import logging
import uuid
import time
import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

import config

# --- Inisialisasi Koneksi MongoDB ---
client = None
db = None

try:
    client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    db = client[config.MONGO_DB_NAME]
    logging.info(f"Successfully connected to MongoDB. Database: '{config.MONGO_DB_NAME}'")
except ConnectionFailure as e:
    logging.critical(f"MongoDB connection failed: {e}", exc_info=True)
    db = None 

def save_detection_history(person_data: dict, image_url: str):
    """
    Saves the AI detection result to the 'history_ai' collection in MongoDB.
    
    Args:
        person_data: A dictionary containing the detected person's information.
        image_url: The URL of the captured image stored on FTP.
    """
    # [PERBAIKAN] Mengubah cara memeriksa koneksi database
    if db is None:
        logging.error("Database not connected. Cannot save history.")
        return

    try:
        history_collection = db.history_ai

        # Mapping data dari deteksi ke skema HistoryAi
        fatigue_level = 1 if person_data.get('fatigue') == 'Drowsy' else 0

        # Membuat dokumen baru
        history_document = {
            'nama': person_data.get('name', 'Tidak Dikenal'),
            'gambar': image_url,
            'mood': person_data.get('emotion', 'N/A'),
            'keletihan': fatigue_level,
            'status_absen': 'Terdeteksi', # Nilai default, bisa disesuaikan
            'userGuid': person_data.get('guid', 'N/A'),
            'guid': str(uuid.uuid4()), # Membuat GUID unik untuk setiap record histori
            'guid_device': config.CAMERA_GUID,
            'datetime': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'timestamp': int(time.time()),
            'unit': person_data.get('unit', 'N/A'),
            'process': 'AI Face Detection',
            'jam_masuk': None,
            'jam_keluar': None,
            'jam_masuk_actual': None,
            'jam_keluar_actual': None,
            'jumlah_telat': 0,
            'total_jam_telat': 0,
            'createdAt': datetime.datetime.now(datetime.timezone.utc),
            'updatedAt': datetime.datetime.now(datetime.timezone.utc),
        }

        result = history_collection.insert_one(history_document)
        logging.info(f"Successfully saved AI detection history to MongoDB with ID: {result.inserted_id}")

    except OperationFailure as e:
        logging.error(f"MongoDB operation failed: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving to MongoDB: {e}", exc_info=True)
