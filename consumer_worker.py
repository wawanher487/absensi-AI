import os
from datetime import datetime
from services.static_service import simpan_dari_path


def proses_deteksi_gambar(local_image_path, user_guid):
    # Simpan ke folder static/detections dan ambil URL-nya
    local_image_url = simpan_dari_path(local_image_path, user_guid)


    # Lakukan sesuatu dengan URL ini
    print("Gambar tersedia di:", local_image_url)

    return local_image_url
