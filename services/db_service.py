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

# Zona waktu Indonesia Barat (WIB = UTC+7)
WIB = datetime.timezone(datetime.timedelta(hours=7))

# Jadwal default
JADWAL_MASUK = "09:00:00"
JADWAL_KELUAR = "17:00:00"

try:
    client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ismaster')
    db = client[config.MONGO_DB_NAME]
    logging.info(f"Successfully connected to MongoDB. Database: '{config.MONGO_DB_NAME}'")
except ConnectionFailure as e:
    logging.critical(f"MongoDB connection failed: {e}", exc_info=True)
    db = None

# --- Data Contoh untuk Absen Masuk ---
data_absen_masuk = {
    'gambar': 'url_gambar_masuk.jpg',
    'mood': 'Marah',
    'keletihan': 16.95,
    'mood_scores': {
        'Marah': 45.51,
        'Senang': 0.72,
        'Netral': 34.58,
        'Sedih': 19.19
    },
    'guid': 'user123',
    'nama': 'Budi Santoso',
    'unit': 'IT'
}

# --- Data Contoh untuk Absen Pulang ---
data_absen_pulang = {
    'gambar': 'url_gambar_pulang.jpg',
    'mood': 'Marah',
    'keletihan': 16.95,
    'mood_scores': {
        'Marah': 45.51,
        'Senang': 0.72,
        'Netral': 34.58,
        'Sedih': 19.19
    },
    'guid': 'user123',
    'nama': 'Budi Santoso',
    'unit': 'IT'
}



# --- Fungsi Baru untuk Mencatat Data Mentah ---
def log_raw_detection_data(person_data: dict, image_url: str):
    """
    Menyimpan data deteksi AI mentah ke koleksi 'raw_detection_logs'.
    Menggunakan zona waktu Indonesia Barat (WIB) untuk semua timestamp.
    """
    if db is None:
        logging.error("Database tidak terhubung. Tidak dapat mencatat data deteksi mentah.")
        return

    try:
        raw_logs_collection = db.raw_detection_logs
        current_time = datetime.datetime.now(WIB)  # Gunakan waktu WIB
        
        log_document = {
            'nama': person_data.get('nama', 'Tidak Dikenal'),
            'gambar': image_url,
            'mood': person_data.get('mood', 'N/A'),
            'keletihan': float(person_data.get('keletihan', 0.0)),
            'Marah': person_data.get('mood_scores', {}).get('Marah', 0.0),
            "Senang": person_data.get('mood_scores', {}).get('Senang', 0.0),
            "Netral": person_data.get('mood_scores', {}).get('Netral', 0.0),
            "Sedih": person_data.get('mood_scores', {}).get('Sedih', 0.0),
            'status_absen': 'Terdeteksi',
            'userGuid': person_data.get('guid', 'N/A'),
            'guid': str(uuid.uuid4()),
            'guid_device': config.CAMERA_GUID,
            'datetime': current_time.isoformat(),
            'timestamp': int(time.time()),
            'unit': person_data.get('unit', 'N/A'),
            'process': 'AI Face Detection Log',
            'jam_masuk':JADWAL_MASUK, # Jadwal masuk default
            'jam_keluar': JADWAL_KELUAR, # Jadwal keluar default
            'jam_masuk_actual':  current_time.strftime('%H:%M:%S'),
            'jam_keluar_actual': None,
            'jumlah_telat': 0,
            'total_jam_telat': "00:00",
            'createdAt': current_time,
            'updatedAt': current_time,
            'raw_input_data': person_data
        }

        result = raw_logs_collection.insert_one(log_document)
        logging.debug(f"Data mentah tercatat. ID: {result.inserted_id}")

    except OperationFailure as e:
        logging.error(f"Operasi MongoDB gagal: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Kesalahan tak terduga: {e}", exc_info=True)


def save_detection_history(person_data: dict, image_url: str):
    """
    Menyimpan/perbarui hasil deteksi AI di koleksi 'history_ai'.
    Menggunakan zona waktu Indonesia Barat (WIB) untuk semua timestamp.
    Sekarang menyimpan status_absen sebagai array ["Hadir", "Pulang"].
    """
    if db is None:
        logging.error("Database tidak terhubung. Operasi dibatalkan.")
        return

    # Catat data mentah terlebih dahulu
    log_raw_detection_data(person_data, image_url)
    
    try:
        history_collection = db.history_ai
        current_time = datetime.datetime.now(WIB)
        today_date_str = current_time.strftime('%Y-%m-%d')
        
        time_str = current_time.strftime('%H:%M:%S')  # HH:MM:SS
        jam_masuk_default = datetime.datetime.strptime(JADWAL_MASUK, "%H:%M").time()
        jam_keluar_default = datetime.datetime.strptime(JADWAL_KELUAR, "%H:%M").time()
        jam_sekarang = current_time.time()

        status_absen = "-"
        jumlah_telat = 0
        total_jam_telat = 0

        # Cek apakah sudah ada record hari ini
        query = {
            'userGuid': person_data.get('guid', 'N/A'),
            'datetime': {'$regex': f'^{today_date_str}'}
        }
        existing_record = history_collection.find_one(query)

        if not existing_record:
            # Belum ada: Absen masuk
            if jam_sekarang <= jam_masuk_default:
                status_absen = "Hadir"
                jumlah_telat = 0
                total_jam_telat = 0
            else:
                status_absen = "Terlambat"
                jumlah_telat = 1
                selisih = datetime.datetime.combine(current_time.date(), jam_sekarang) - datetime.datetime.combine(current_time.date(), jam_masuk_default)
                total_jam_telat = int(selisih.total_seconds() // 60)

            # Simpan record baru
            logging.info("Membuat record baru")
            history_document = {
                'nama': person_data.get('nama', 'Tidak Dikenal'),
                'gambar': image_url,
                'mood': person_data.get('mood', 'N/A'),
                'keletihan': float(person_data.get('keletihan', 0.0)),
                'Marah': person_data.get('mood_scores', {}).get('Marah', 0.0),
                "Senang": person_data.get('mood_scores', {}).get('Senang', 0.0),
                "Netral": person_data.get('mood_scores', {}).get('Netral', 0.0),
                "Sedih": person_data.get('mood_scores', {}).get('Sedih', 0.0),
                'status_absen': [status_absen],  # ⬅️ Langsung array
                'userGuid': person_data.get('guid', 'N/A'),
                'guid': str(uuid.uuid4()),
                'guid_device': config.CAMERA_GUID,
                'datetime': current_time.isoformat(),
                'timestamp': int(time.time()),
                'unit': person_data.get('unit', 'N/A'),
                'process': 'done',
                'jam_masuk': JADWAL_MASUK,
                'jam_keluar': JADWAL_KELUAR,
                'jam_masuk_actual': time_str,
                'jam_keluar_actual': None,
                'jumlah_telat': jumlah_telat,
                'total_jam_telat': total_jam_telat,
                'createdAt': current_time,
                'updatedAt': current_time
            }
            result = history_collection.insert_one(history_document)
            logging.info(f"Record baru dibuat. ID: {result.inserted_id}")

        else:
            # Sudah ada: Absen pulang
            if jam_sekarang < jam_keluar_default:
                logging.info(f"Belum waktunya absen pulang untuk {person_data.get('nama')}, sekarang jam {jam_sekarang.strftime('%H:%M:%S')}")
                return  #  Belum waktunya pulang

            status_absen = "Pulang"

            logging.info(f"Memperbarui record untuk {person_data.get('nama')} sebagai pulang")

            status_lama = existing_record.get('status_absen', [])
            status_baru = list(set(status_lama + [status_absen]))  # ⬅️ Tambah "Pulang" jika belum ada

            update_fields = {
                '$set': {
                    'gambar': image_url,
                    'mood': person_data.get('mood', 'N/A'),
                    'keletihan': float(person_data.get('keletihan', 0.0)),
                    'Marah': person_data.get('mood_scores', {}).get('Marah', 0.0),
                    "Senang": person_data.get('mood_scores', {}).get('Senang', 0.0),
                    "Netral": person_data.get('mood_scores', {}).get('Netral', 0.0),
                    "Sedih": person_data.get('mood_scores', {}).get('Sedih', 0.0),
                    'status_absen': status_baru,  # ⬅️ Array gabungan
                    'unit': person_data.get('unit', 'N/A'),
                    'datetime': current_time.isoformat(),
                    'timestamp': int(time.time()),
                    'jam_keluar': JADWAL_KELUAR,
                    'jam_keluar_actual': time_str,
                    'updatedAt': current_time
                }
            }

            result = history_collection.update_one(query, update_fields)
            if result.modified_count > 0:
                logging.info(f"Record diperbarui. ID: {existing_record.get('_id')}")
            else:
                logging.warning("Tidak ada perubahan data saat update.")

    except OperationFailure as e:
        logging.error(f"Operasi MongoDB gagal: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Kesalahan tak terduga: {e}", exc_info=True)

