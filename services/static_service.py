# services/static_service.py

import os
from datetime import datetime
import shutil
import config

# services/static_service.py
def simpan_dari_bytes(image_bytes, user_guid, filename: str | None = None):
    folder = os.path.join("static", "detections")
    os.makedirs(folder, exist_ok=True)

    if filename is None:                      
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{user_guid}_{timestamp}.jpg"

    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)

    return f"{filename}"

def simpan_dari_path(local_image_path, user_guid, base_url=config.BASE_URL):
    """
    Simpan gambar dari path lokal (digunakan di consumer_worker.py)
    """
    folder = os.path.join("static", "detections")
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_{user_guid}_{timestamp}.jpg"
    target_path = os.path.join(folder, filename)

    shutil.copy(local_image_path, target_path)

    return f"{filename}"
