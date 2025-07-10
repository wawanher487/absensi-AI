import logging
import requests
import json
import config

def setup_logging():
    """Mengonfigurasi logger root untuk aplikasi."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_and_map_users_from_api():
    """
    [DIUBAH] Mengambil data pengguna dari API dengan fallback ke file statis.
    - Mencoba mengambil dari API terlebih dahulu.
    - Jika berhasil, simpan data ke file statis untuk penggunaan di masa mendatang.
    - Jika API gagal, coba muat dari file statis.
    """
    # 1. Coba ambil dari API
    logging.info("Mencoba mengambil data pengguna dari API...")
    try:
        response = requests.get(config.API_URL, timeout=10)
        response.raise_for_status()  # Akan memunculkan error untuk status 4xx/5xx
        users_list = response.json().get('data', [])

        if users_list:
            logging.info(f"Berhasil mengambil {len(users_list)} pengguna dari API.")
            
            # Simpan data baru ke file statis. Pastikan STATIC_USERS_PATH ada di config.py
            try:
                with open(config.STATIC_USERS_PATH, 'w', encoding='utf-8') as f:
                    json.dump(users_list, f, ensure_ascii=False, indent=4)
                logging.info(f"Berhasil menyimpan data pengguna ke file statis: {config.STATIC_USERS_PATH}")
            except IOError as e:
                logging.error(f"Tidak dapat menulis ke file pengguna statis: {e}")

            # Petakan data dan kembalikan hasilnya
            # user_map = {}
            # for user in users_list:
            #     user_key = f"{user['name'].replace(' ', '_')}_{user['guid']}"
            #     user_map[user_key] = user
            # logging.info(f"Berhasil memetakan {len(user_map)} pengguna.")
            # return user_map
            user_map = {}
            for user in users_list:
                # Kunci HARUS hanya GUID-nya saja, dan pastikan bersih dari spasi
                user_key = str(user['guid']).strip() # <-- Kunci yang Benar
                user_map[user_key] = user
            return user_map
        else:
            logging.warning("API tidak mengembalikan data pengguna.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Gagal mengambil data pengguna dari API: {e}. Mencoba memuat dari file statis.")
    
    # 2. Jika API gagal atau tidak mengembalikan apa pun, gunakan file statis sebagai fallback
    logging.info(f"Menggunakan fallback ke file pengguna statis: {config.STATIC_USERS_PATH}")
    try:
        with open(config.STATIC_USERS_PATH, 'r', encoding='utf-8') as f:
            users_list = json.load(f)
        
        logging.info(f"Berhasil memuat {len(users_list)} pengguna dari file statis.")
        
        user_map = {}
        for user in users_list:
            user_key = f"{user['nama'].replace(' ', '_')}_{user['guid']}"
            user_map[user_key] = user
        return user_map

    except FileNotFoundError:
        logging.error("File pengguna statis tidak ditemukan. Tidak ada pengguna yang dapat dimuat.")
        return {}
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Gagal membaca atau mem-parsing file pengguna statis: {e}")
        return {}
