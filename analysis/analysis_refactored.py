# analysis/analysis_refactored.py

import os
import cv2
import numpy as np
import dlib
import face_recognition
import joblib
import logging
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from typing import List, Dict, Tuple, Any, Optional

import config

class FaceAnalysisSystem:
    """
    Kelas terpusat untuk melakukan analisis wajah: identifikasi, keletihan, dan emosi.
    Mengelola pemuatan dan penggunaan semua model yang diperlukan.
    """

    def __init__(self):
        """Inisialisasi sistem dan langsung memuat semua model."""
        self.detector: Optional[dlib.fhog_object_detector] = None
        self.predictor: Optional[dlib.shape_predictor] = None
        self.emotion_classifier: Optional[any] = None
        self.knn_clf: Optional[KNeighborsClassifier] = None
        self._load_models()

    def _load_models(self):
        """Memuat semua model yang diperlukan (Dlib, Emotion, Face Recognition) ke memori."""
        logging.info("Memulai pemuatan semua model ke dalam memori...")
        try:
            # [INFO] Model-model ini wajib ada agar sistem bisa berjalan.
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(config.DLIB_SHAPE_PREDICTOR)
            self.emotion_classifier = load_model(config.EMOTION_MODEL_PATH, compile=False)
            logging.info("Model Dlib (detector, predictor) dan Keras (emotion) berhasil dimuat.")
            
            # [INFO] Model k-NN bersifat opsional; bisa dibuat nanti melalui training.
            if os.path.exists(config.MODEL_PATH):
                self.knn_clf = joblib.load(config.MODEL_PATH)
                logging.info(f"Model k-NN (identifikasi) berhasil dimuat dari: {config.MODEL_PATH}")
            else:
                logging.warning(f"File model k-NN tidak ditemukan di '{config.MODEL_PATH}'. Fungsi identifikasi tidak akan akurat sampai training dijalankan.")
        except Exception as e:
            # [PERBAIKAN] Logging dibuat lebih spesifik untuk menunjukkan kegagalan kritis.
            logging.critical(f"GAGAL memuat model penting: {e}", exc_info=True)
            # Reset semua model untuk memastikan state konsisten (semua None).
            self.detector, self.predictor, self.emotion_classifier, self.knn_clf = None, None, None, None
            
    def is_ready(self) -> bool:
        """
        Memeriksa apakah semua model PENTING (detector, predictor, emotion)
        telah berhasil dimuat. Model k-NN opsional.
        """
        # [PERBAIKAN] Dibuat lebih eksplisit model mana yang wajib.
        return all([self.detector, self.predictor, self.emotion_classifier])

    @staticmethod
    def _calculate_ear(eye: np.ndarray) -> float:
        """Menghitung Eye Aspect Ratio (EAR) untuk satu mata."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

    def analyze_image(self, image: np.ndarray, user_details_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Menganalisis satu frame gambar untuk mendeteksi semua wajah dan mengekstrak
        informasi identitas, keletihan, dan mood untuk setiap wajah.
        """
        if not self.is_ready():
            logging.error("Sistem analisis tidak siap (model penting tidak dimuat). Analisis dibatalkan.")
            return []
        # print(user_details_map)  # Debugging: Tampilkan isi user_details_map
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # face_recognition bekerja lebih baik dengan RGB
        face_rects = self.detector(gray_image, 0)
        results = []

        # [INFO] Jika face_rects kosong, loop tidak akan berjalan dan fungsi mengembalikan list kosong.
        # Ini adalah perilaku yang diharapkan.
        for rect in face_rects:
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            # Format (top, right, bottom, left) untuk face_recognition
            face_box_fr = (y, x + w, y + h, x)

            # --- 1. Analisis Identitas (k-NN) ---
            nama, guid, unit = "Tidak Dikenal", None, "-"
            if self.knn_clf:
                # [INFO] Mencari encoding hanya untuk area wajah yang terdeteksi
                encodings = face_recognition.face_encodings(rgb_image, [face_box_fr])
                if encodings:
                    closest_distances = self.knn_clf.kneighbors(encodings, n_neighbors=1)
                    # --- TAMBAHKAN BARIS INI UNTUK DEBUGGING ---
                    predicted_label_guid = self.knn_clf.predict(encodings)[0]
                    distance_value = closest_distances[0][0][0]
                    logging.info(f"--> DEBUG: Prediksi GUID: {predicted_label_guid}, Jarak: {distance_value:.4f}, Threshold: {config.KNN_DISTANCE_THRESHOLD}")
                    # Di dalam metode analyze_image, dalam blok if...
                    # ...
                    # predicted_label_guid = self.knn_clf.predict(encodings)[0]

                    # --- [ULTIMATE DEBUG] GANTI BLOK DEBUG ANDA DENGAN INI ---
                    logging.info("--- ULTIMATE DEBUG ---")
                    logging.info(f"PREDICTED GUID: '{predicted_label_guid}' (Tipe: {type(predicted_label_guid).__name__})")

                    # Cek tipe data dari KUNCI PERTAMA di map
                    if user_details_map:
                        first_key = list(user_details_map.keys())[0]
                        logging.info(f"FIRST MAP KEY:  '{first_key}' (Tipe: {type(first_key).__name__})")
                    else:
                        logging.info("USER DETAILS MAP KOSONG!")

                    logging.info("----------------------")
                    # --------------------------------------------------------

                    user_info = user_details_map.get(predicted_label_guid, {})
                    # ...
                    # -------------------------------------------
                    # [INFO] Pemeriksaan threshold adalah kunci dari pengenalan
                    # if closest_distances[0][0][0] <= config.KNN_DISTANCE_THRESHOLD:
                    #     predicted_label_guid = self.knn_clf.predict(encodings)[0]
                    #     print(f"Prediksi GUID: {predicted_label_guid}, Jarak: {distance_value:.4f}, Threshold: {config.KNN_DISTANCE_THRESHOLD}")
                    #     # Ambil detail pengguna dari map yang sudah di-cache
                    #     user_info = user_details_map.get(predicted_label_guid, {})
                    #     nama = user_info.get("name", "Nama Tidak Ditemukan di Map")
                    #     guid = user_info.get("guid") # guid akan sama dengan predicted_label_guid
                    #     unit = user_info.get("unit", "-")
                    # ...
                    if closest_distances[0][0][0] <= config.KNN_DISTANCE_THRESHOLD:
                        # [PERBAIKAN FINAL] Bersihkan hasil prediksi dari spasi
                        predicted_label_guid = self.knn_clf.predict(encodings)[0].strip()
                        logging.info(f"Prediksi GUID: {predicted_label_guid}, Jarak: {closest_distances[0][0][0]:.4f}, Threshold: {config.KNN_DISTANCE_THRESHOLD}")
                        # Sekarang pencarian akan berhasil
                        user_info = user_details_map.get(predicted_label_guid, {})
                        nama = user_info.get("name", "Nama Tidak Ditemukan di Map")
                        guid = user_info.get("guid")
                        unit = user_info.get("unit", "-")
# ...

            # --- 2. Analisis Keletihan & Mood ---
            shape = self.predictor(gray_image, rect)
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            # Keletihan (Fatigue) berdasarkan Eye Aspect Ratio (EAR)
            left_ear = self._calculate_ear(landmarks[config.LEFT_EYE_INDICES])
            right_ear = self._calculate_ear(landmarks[config.RIGHT_EYE_INDICES])
            avg_ear = (left_ear + right_ear) / 2.0
            fatigue_level = np.interp(avg_ear, [config.EAR_THRESHOLD_CLOSED, config.EAR_THRESHOLD_OPEN], [100, 0])
            keletihan = round(np.clip(fatigue_level, 0, 100), 2)

            # Mood (Emotion) berdasarkan model Keras
            mood = "N/A"
            mood_scores = {label: 0.0 for label in config.EMOTION_LABELS_INDONESIAN}
            roi_gray = gray_image[y:y + h, x:x + w]
            if roi_gray.size > 0:
                roi_resized = cv2.resize(roi_gray, config.EMOTION_TARGET_SIZE)
                # [PERBAIKAN] Ubah tipe data secara eksplisit ke float32
                roi_normalized = roi_resized.astype('float32') / 255.0
                roi_final = np.expand_dims(np.expand_dims(roi_normalized, axis=-1), axis=0)
                predictions = self.emotion_classifier.predict(roi_final, verbose=0)[0]
            # ...
                
                all_moods_en = {label: float(p) for label, p in zip(config.EMOTION_LABELS_ENGLISH, predictions)}
                dominant_mood_en = max(all_moods_en, key=all_moods_en.get)
                mood = config.EMOTION_MAP_TO_INDONESIAN.get(dominant_mood_en, "N/A")
                
                # Mengisi skor mood yang relevan untuk dikirim ke frontend
                for en_label, id_label in config.EMOTION_MAP_TO_INDONESIAN.items():
                     if id_label in mood_scores:
                        mood_scores[id_label] = round(all_moods_en.get(en_label, 0.0) * 100, 2)
            
            results.append({
                "box": (x, y, w, h), "nama": nama, "guid": guid, "unit": unit,
                "mood": mood, "keletihan": keletihan, "mood_scores": mood_scores
            })

        return results

    def train_model(self) -> Tuple[bool, str]:
        """Melatih model k-NN dari dataset dan menyimpannya ke file."""
        logging.info("Memulai proses training model k-NN...")
        encodings, labels = [], []
        
        if not os.path.isdir(config.DATASET_PATH):
            msg = f"Direktori dataset tidak ditemukan di: {config.DATASET_PATH}"
            logging.error(msg)
            return False, msg

        # [INFO] Iterasi melalui folder pengguna (contoh: 'NamaPengguna_guid123')
        for user_folder in os.listdir(config.DATASET_PATH):
            user_path = os.path.join(config.DATASET_PATH, user_folder)
            if not os.path.isdir(user_path):
                continue
            
            try:
                # Ekstrak GUID dari nama folder
                # guid_label = user_folder.split('_')[-1]
                guid_label = user_folder.split('_')[-1].strip() 
            except IndexError:
                logging.warning(f"Folder '{user_folder}' tidak sesuai format 'Nama_GUID'. Dilewati.")
                continue

            image_count_per_user = 0
            for img_name in os.listdir(user_path):
                img_path = os.path.join(user_path, img_name)
                try:
                    image = face_recognition.load_image_file(img_path)
                    face_encs = face_recognition.face_encodings(image)
                    if face_encs:
                        # [INFO] Hanya gunakan encoding wajah pertama yang ditemukan di gambar
                        encodings.append(face_encs[0])
                        labels.append(guid_label)
                        image_count_per_user += 1
                    else:
                        logging.warning(f"Tidak ada wajah yang ditemukan di {img_path}")
                except Exception as e:
                    logging.error(f"Gagal memproses gambar {img_path}: {e}")
            logging.info(f"Memproses {image_count_per_user} gambar untuk GUID: {guid_label}")
        
        if not encodings:
            msg = "Training gagal, tidak ada wajah yang berhasil di-encode dari dataset."
            logging.error(msg)
            return False, msg

        # [INFO] Menentukan jumlah tetangga k-NN secara dinamis
        n_neighbors = max(1, int(round(np.sqrt(len(np.unique(labels))))))
        logging.info(f"Melatih model k-NN dengan n_neighbors = {n_neighbors}")
        
        new_clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
        new_clf.fit(encodings, labels)
        
        joblib.dump(new_clf, config.MODEL_PATH)
        
        # [PERBAIKAN] Langsung perbarui model yang ada di memori tanpa perlu restart.
        self.knn_clf = new_clf
        
        msg = f"Training berhasil. Model dilatih dengan {len(labels)} gambar dari {len(np.unique(labels))} orang. Disimpan di {config.MODEL_PATH}"
        logging.info(msg)
        return True, msg