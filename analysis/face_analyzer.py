import os
import cv2
import numpy as np
import dlib
import face_recognition
import joblib
import pickle
import logging
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier

import config

# --- Module-level state for models ---
# These are loaded once when load_models() is called.
detector = None
predictor = None
emotion_classifier = None
knn_clf = None

def load_models():
    """Memuat semua model yang diperlukan (Dlib, Emotion, Face Recognition) ke memori."""
    global detector, predictor, emotion_classifier, knn_clf
    logging.info("Memuat semua model...")

    # Memuat Dlib face detector dan shape predictor
    if os.path.exists(config.DLIB_SHAPE_PREDICTOR):
        try:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(config.DLIB_SHAPE_PREDICTOR)
            logging.info("Dlib face detector dan shape predictor berhasil dimuat.")
        except Exception as e:
            logging.error(f"Error saat memuat model Dlib: {e}")
    else:
        logging.critical(f"Model Dlib '{config.DLIB_SHAPE_PREDICTOR}' tidak ditemukan! Deteksi dan analisis wajah tidak akan berfungsi.")

    # Memuat model deteksi emosi
    if os.path.exists(config.EMOTION_MODEL_PATH):
        try:
            emotion_classifier = load_model(config.EMOTION_MODEL_PATH, compile=False)
            logging.info("Model deteksi emosi berhasil dimuat.")
        except Exception as e:
            logging.error(f"Error saat memuat model deteksi emosi: {e}")
    else:
        logging.critical(f"Model emosi '{config.EMOTION_MODEL_PATH}' tidak ditemukan! Deteksi emosi akan dinonaktifkan.")

    # Memuat model pengenalan wajah k-NN
    if os.path.exists(config.MODEL_PATH):
        try:
            knn_clf = joblib.load(config.MODEL_PATH)
            logging.info("Model pengenalan wajah (k-NN) berhasil dimuat.")
        except (EOFError, pickle.UnpicklingError):
            logging.error(f"File model '{config.MODEL_PATH}' korup. Harap hapus dan latih ulang.")
            knn_clf = None # Pastikan model yang korup tidak digunakan.
        except Exception as e:
            logging.error(f"Error saat memuat model pengenalan wajah: {e}")
    else:
        logging.warning("Model pengenalan wajah tidak ditemukan. Silakan latih model terlebih dahulu.")


def _calculate_eye_aspect_ratio(eye):
    """Menghitung eye aspect ratio (EAR) untuk mendeteksi kedipan/rasa kantuk."""
    # Menghitung jarak euclidean antara dua pasang
    # landmark mata vertikal (koordinat x, y)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Menghitung jarak euclidean antara landmark mata
    # horizontal (koordinat x, y)
    C = dist.euclidean(eye[0], eye[3])
    # Menghitung eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def analyze_image_for_faces(image: np.ndarray, user_details_map: dict) -> list:
    """
    Menganalisis satu frame gambar untuk mendeteksi dan mengidentifikasi wajah, emosi, dan kelelahan.
    
    Args:
        image: Gambar input dalam format BGR (dari OpenCV).
        user_details_map: Dictionary yang memetakan identifier pengguna ke detail mereka.

    Returns:
        Sebuah list berisi dictionary, di mana setiap dictionary berisi detail untuk wajah yang terdeteksi.
    """
    if detector is None:
        logging.warning("Detector Dlib tidak dimuat, tidak dapat menganalisis gambar.")
        return []

    # Konversi ke grayscale untuk Dlib dan siapkan RGB untuk face_recognition
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_rects = detector(gray_image, 0)
    results = []

    for rect in face_rects:
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        face_box = (y, x + w, y + h, x) # format (top, right, bottom, left)

        result = {
            "box": (x, y, w, h),
            "name": "Tidak Dikenal",
            "unit": "-",
            "guid": None,
            "fatigue": "N/A",
            "emotion": "N/A"
        }

        # --- 1. Pengenalan Wajah ---
        if knn_clf:
            encodings = face_recognition.face_encodings(rgb_image, [face_box])
            if encodings:
                # Gunakan model k-NN untuk menemukan kecocokan terbaik
                closest_distances = knn_clf.kneighbors(encodings, n_neighbors=1)
                # Periksa apakah kecocokan berada dalam ambang batas toleransi
                if closest_distances[0][0][0] <= 0.4: # Ambang batas yang lebih ketat
                    predicted_name_guid = knn_clf.predict(encodings)[0]
                    user_info = user_details_map.get(predicted_name_guid)
                    if user_info:
                        # Perbarui hasil dengan detail pengguna yang ditemukan
                        result.update({
                            "name": user_info.get("name", "Tidak Dikenal"),
                            "unit": user_info.get("unit", "-"),
                            "guid": user_info.get("guid")
                        })

        # --- 2. Deteksi Kelelahan (Rasa Kantuk) ---
        if predictor:
            shape = predictor(gray_image, rect)
            shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            # Ekstrak landmark untuk mata kiri dan kanan
            left_eye = shape_np[42:48]
            right_eye = shape_np[36:42]
            
            left_ear = _calculate_eye_aspect_ratio(left_eye)
            right_ear = _calculate_eye_aspect_ratio(right_eye)
            
            avg_ear = (left_ear + right_ear) / 2.0
            result["fatigue"] = "Mengantuk" if avg_ear < 0.25 else "Sadar"

        # --- 3. Deteksi Emosi ---
        if emotion_classifier:
            # Ekstrak Region of Interest (ROI) untuk model emosi
            roi_gray = gray_image[y:y + h, x:x + w]
            # Lakukan pra-pemrosesan ROI agar sesuai dengan kebutuhan input model
            roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_normalized = roi_resized.astype('float') / 255.0
            roi_expanded = np.expand_dims(roi_normalized, axis=-1) # Tambahkan dimensi channel
            roi_final = np.expand_dims(roi_expanded, axis=0)      # Tambahkan dimensi batch
            
            # Prediksi emosi
            prediction = emotion_classifier.predict(roi_final, verbose=0)[0]
            result["emotion"] = config.EMOTION_LABELS[prediction.argmax()]

        results.append(result)

    return results


def train_model() -> (bool, str):
    """
    Melatih model pengenalan wajah k-NN pada dataset.
    
    Returns:
        Sebuah tuple (success, message).
    """
    global knn_clf
    logging.info("================== PROSES TRAINING DIMULAI ==================")
    face_encodings, labels = [], []

    if not os.path.exists(config.DATASET_PATH):
        logging.error("Direktori dataset tidak ditemukan.")
        return False, "Direktori dataset tidak ditemukan."

    logging.info(f"Mulai memindai direktori dataset di: {config.DATASET_PATH}")
    # Iterasi melalui setiap orang di dataset
    for user_folder in os.listdir(config.DATASET_PATH):
        user_path = os.path.join(config.DATASET_PATH, user_folder)
        if not os.path.isdir(user_path):
            continue
        
        logging.info(f"--- Memproses folder pengguna: {user_folder} ---")
        # Iterasi melalui setiap gambar training untuk orang saat ini
        for image_name in os.listdir(user_path):
            image_path = os.path.join(user_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                logging.info(f"  -> Memproses file gambar: {image_name}")
                image = face_recognition.load_image_file(image_path)
                # Temukan semua lokasi wajah di gambar
                face_locations = face_recognition.face_locations(image)
                
                # Asumsikan hanya satu wajah per gambar training, dapatkan encoding-nya
                if face_locations:
                    logging.info(f"     * Wajah ditemukan. Mengekstrak encoding...")
                    encodings = face_recognition.face_encodings(image, face_locations)
                    if encodings:
                        face_encodings.append(encodings[0])
                        labels.append(user_folder)
                        logging.info(f"     * Encoding berhasil diekstrak dan ditambahkan.")
                else:
                    logging.warning(f"     * TIDAK ADA WAJAH ditemukan di {image_name}, dilewati.")
            except Exception as e:
                logging.error(f"  -> Gagal memproses gambar {image_path}: {e}")

    if not face_encodings:
        logging.error("Training GAGAL: Tidak ada wajah yang dapat di-encode dari seluruh dataset.")
        return False, "Training gagal: Tidak ada wajah yang ditemukan di dataset."

    logging.info("---------------------------------------------------------")
    logging.info(f"Total {len(face_encodings)} enconding wajah berhasil dikumpulkan.")
    
    # Tentukan jumlah tetangga yang optimal untuk k-NN
    n_neighbors = int(round(np.sqrt(len(face_encodings))))
    logging.info(f"Mempersiapkan training model k-NN dengan n_neighbors = {n_neighbors}.")
    
    new_knn_clf = KNeighborsClassifier(n_neighbors=max(1, n_neighbors), algorithm='ball_tree', weights='distance')
    
    logging.info("Memulai fitting model k-NN dengan data encoding...")
    new_knn_clf.fit(face_encodings, labels)
    logging.info("Fitting model k-NN selesai.")

    # Simpan model yang telah dilatih ke file
    logging.info(f"Menyimpan model yang sudah dilatih ke file: {config.MODEL_PATH}")
    joblib.dump(new_knn_clf, config.MODEL_PATH)
    
    # Segera perbarui model di memori
    knn_clf = new_knn_clf
    logging.info("Model di memori telah diperbarui dengan versi yang baru.")
    logging.info("================== PROSES TRAINING SELESAI ==================")
    
    return True, "Model telah berhasil dilatih ulang dan diperbarui!"
