# # import os
# # import cv2
# # import numpy as np
# # import dlib
# # import face_recognition
# # import joblib
# # import pickle
# # import logging
# # from scipy.spatial import distance as dist
# # from tensorflow.keras.models import load_model
# # from sklearn.neighbors import KNeighborsClassifier

# # import config

# # # --- Module-level state for models ---
# # # These are loaded once when load_models() is called.
# # detector = None
# # predictor = None
# # emotion_classifier = None
# # knn_clf = None

# # def load_models():
# #     """Memuat semua model yang diperlukan (Dlib, Emotion, Face Recognition) ke memori."""
# #     global detector, predictor, emotion_classifier, knn_clf
# #     logging.info("Memuat semua model...")

# #     # Memuat Dlib face detector dan shape predictor
# #     if os.path.exists(config.DLIB_SHAPE_PREDICTOR):
# #         try:
# #             detector = dlib.get_frontal_face_detector()
# #             predictor = dlib.shape_predictor(config.DLIB_SHAPE_PREDICTOR)
# #             logging.info("Dlib face detector dan shape predictor berhasil dimuat.")
# #         except Exception as e:
# #             logging.error(f"Error saat memuat model Dlib: {e}")
# #     else:
# #         logging.critical(f"Model Dlib '{config.DLIB_SHAPE_PREDICTOR}' tidak ditemukan! Deteksi dan analisis wajah tidak akan berfungsi.")

# #     # Memuat model deteksi emosi
# #     if os.path.exists(config.EMOTION_MODEL_PATH):
# #         try:
# #             emotion_classifier = load_model(config.EMOTION_MODEL_PATH, compile=False)
# #             logging.info("Model deteksi emosi berhasil dimuat.")
# #         except Exception as e:
# #             logging.error(f"Error saat memuat model deteksi emosi: {e}")
# #     else:
# #         logging.critical(f"Model emosi '{config.EMOTION_MODEL_PATH}' tidak ditemukan! Deteksi emosi akan dinonaktifkan.")

# #     # Memuat model pengenalan wajah k-NN
# #     if os.path.exists(config.MODEL_PATH):
# #         try:
# #             knn_clf = joblib.load(config.MODEL_PATH)
# #             logging.info("Model pengenalan wajah (k-NN) berhasil dimuat.")
# #         except (EOFError, pickle.UnpicklingError):
# #             logging.error(f"File model '{config.MODEL_PATH}' korup. Harap hapus dan latih ulang.")
# #             knn_clf = None # Pastikan model yang korup tidak digunakan.
# #         except Exception as e:
# #             logging.error(f"Error saat memuat model pengenalan wajah: {e}")
# #     else:
# #         logging.warning("Model pengenalan wajah tidak ditemukan. Silakan latih model terlebih dahulu.")


# # def _calculate_eye_aspect_ratio(eye):
# #     """Menghitung eye aspect ratio (EAR) untuk mendeteksi kedipan/rasa kantuk."""
# #     # Menghitung jarak euclidean antara dua pasang
# #     # landmark mata vertikal (koordinat x, y)
# #     A = dist.euclidean(eye[1], eye[5])
# #     B = dist.euclidean(eye[2], eye[4])
# #     # Menghitung jarak euclidean antara landmark mata
# #     # horizontal (koordinat x, y)
# #     C = dist.euclidean(eye[0], eye[3])
# #     # Menghitung eye aspect ratio
# #     ear = (A + B) / (2.0 * C)
# #     return ear

# # def analyze_image_for_faces(image: np.ndarray, user_details_map: dict) -> list:
# #     """
# #     Menganalisis satu frame gambar untuk mendeteksi dan mengidentifikasi wajah, emosi, dan kelelahan.
    
# #     Args:
# #         image: Gambar input dalam format BGR (dari OpenCV).
# #         user_details_map: Dictionary yang memetakan identifier pengguna ke detail mereka.

# #     Returns:
# #         Sebuah list berisi dictionary, di mana setiap dictionary berisi detail untuk wajah yang terdeteksi.
# #     """
# #     if detector is None:
# #         logging.warning("Detector Dlib tidak dimuat, tidak dapat menganalisis gambar.")
# #         return []

# #     # Konversi ke grayscale untuk Dlib dan siapkan RGB untuk face_recognition
# #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
# #     face_rects = detector(gray_image, 0)
# #     results = []

# #     for rect in face_rects:
# #         (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
# #         face_box = (y, x + w, y + h, x) # format (top, right, bottom, left)

# #         result = {
# #             "box": (x, y, w, h),
# #             "name": "Tidak Dikenal",
# #             "unit": "-",
# #             "guid": None,
# #             "fatigue": "N/A",
# #             "emotion": "N/A"
# #         }

# #         # --- 1. Pengenalan Wajah ---
# #         if knn_clf:
# #             encodings = face_recognition.face_encodings(rgb_image, [face_box])
# #             if encodings:
# #                 # Gunakan model k-NN untuk menemukan kecocokan terbaik
# #                 closest_distances = knn_clf.kneighbors(encodings, n_neighbors=1)
# #                 # Periksa apakah kecocokan berada dalam ambang batas toleransi
# #                 if closest_distances[0][0][0] <= 0.4: # Ambang batas yang lebih ketat
# #                     predicted_name_guid = knn_clf.predict(encodings)[0]
# #                     user_info = user_details_map.get(predicted_name_guid)
# #                     if user_info:
# #                         # Perbarui hasil dengan detail pengguna yang ditemukan
# #                         result.update({
# #                             "name": user_info.get("name", "Tidak Dikenal"),
# #                             "unit": user_info.get("unit", "-"),
# #                             "guid": user_info.get("guid")
# #                         })

# #         # --- 2. Deteksi Kelelahan (Rasa Kantuk) ---
# #         if predictor:
# #             shape = predictor(gray_image, rect)
# #             shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
# #             # Ekstrak landmark untuk mata kiri dan kanan
# #             left_eye = shape_np[42:48]
# #             right_eye = shape_np[36:42]
            
# #             left_ear = _calculate_eye_aspect_ratio(left_eye)
# #             right_ear = _calculate_eye_aspect_ratio(right_eye)
            
# #             avg_ear = (left_ear + right_ear) / 2.0
# #             result["fatigue"] = "Mengantuk" if avg_ear < 0.25 else "Sadar"

# #         # --- 3. Deteksi Emosi ---
# #         if emotion_classifier:
# #             # Ekstrak Region of Interest (ROI) untuk model emosi
# #             roi_gray = gray_image[y:y + h, x:x + w]
# #             # Lakukan pra-pemrosesan ROI agar sesuai dengan kebutuhan input model
# #             roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
# #             roi_normalized = roi_resized.astype('float') / 255.0
# #             roi_expanded = np.expand_dims(roi_normalized, axis=-1) # Tambahkan dimensi channel
# #             roi_final = np.expand_dims(roi_expanded, axis=0)      # Tambahkan dimensi batch
            
# #             # Prediksi emosi
# #             prediction = emotion_classifier.predict(roi_final, verbose=0)[0]
# #             result["emotion"] = config.EMOTION_LABELS[prediction.argmax()]

# #         results.append(result)

# #     return results


# # def train_model() -> (bool, str):
# #     """
# #     Melatih model pengenalan wajah k-NN pada dataset.
    
# #     Returns:
# #         Sebuah tuple (success, message).
# #     """
# #     global knn_clf
# #     logging.info("================== PROSES TRAINING DIMULAI ==================")
# #     face_encodings, labels = [], []

# #     if not os.path.exists(config.DATASET_PATH):
# #         logging.error("Direktori dataset tidak ditemukan.")
# #         return False, "Direktori dataset tidak ditemukan."

# #     logging.info(f"Mulai memindai direktori dataset di: {config.DATASET_PATH}")
# #     # Iterasi melalui setiap orang di dataset
# #     for user_folder in os.listdir(config.DATASET_PATH):
# #         user_path = os.path.join(config.DATASET_PATH, user_folder)
# #         if not os.path.isdir(user_path):
# #             continue
        
# #         logging.info(f"--- Memproses folder pengguna: {user_folder} ---")
# #         # Iterasi melalui setiap gambar training untuk orang saat ini
# #         for image_name in os.listdir(user_path):
# #             image_path = os.path.join(user_path, image_name)
# #             if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
# #                 continue

# #             try:
# #                 logging.info(f"  -> Memproses file gambar: {image_name}")
# #                 image = face_recognition.load_image_file(image_path)
# #                 # Temukan semua lokasi wajah di gambar
# #                 face_locations = face_recognition.face_locations(image)
                
# #                 # Asumsikan hanya satu wajah per gambar training, dapatkan encoding-nya
# #                 if face_locations:
# #                     logging.info(f"     * Wajah ditemukan. Mengekstrak encoding...")
# #                     encodings = face_recognition.face_encodings(image, face_locations)
# #                     if encodings:
# #                         face_encodings.append(encodings[0])
# #                         labels.append(user_folder)
# #                         logging.info(f"     * Encoding berhasil diekstrak dan ditambahkan.")
# #                 else:
# #                     logging.warning(f"     * TIDAK ADA WAJAH ditemukan di {image_name}, dilewati.")
# #             except Exception as e:
# #                 logging.error(f"  -> Gagal memproses gambar {image_path}: {e}")

# #     if not face_encodings:
# #         logging.error("Training GAGAL: Tidak ada wajah yang dapat di-encode dari seluruh dataset.")
# #         return False, "Training gagal: Tidak ada wajah yang ditemukan di dataset."

# #     logging.info("---------------------------------------------------------")
# #     logging.info(f"Total {len(face_encodings)} enconding wajah berhasil dikumpulkan.")
    
# #     # Tentukan jumlah tetangga yang optimal untuk k-NN
# #     n_neighbors = int(round(np.sqrt(len(face_encodings))))
# #     logging.info(f"Mempersiapkan training model k-NN dengan n_neighbors = {n_neighbors}.")
    
# #     new_knn_clf = KNeighborsClassifier(n_neighbors=max(1, n_neighbors), algorithm='ball_tree', weights='distance')
    
# #     logging.info("Memulai fitting model k-NN dengan data encoding...")
# #     new_knn_clf.fit(face_encodings, labels)
# #     logging.info("Fitting model k-NN selesai.")

# #     # Simpan model yang telah dilatih ke file
# #     logging.info(f"Menyimpan model yang sudah dilatih ke file: {config.MODEL_PATH}")
# #     joblib.dump(new_knn_clf, config.MODEL_PATH)
    
# #     # Segera perbarui model di memori
# #     knn_clf = new_knn_clf
# #     logging.info("Model di memori telah diperbarui dengan versi yang baru.")
# #     logging.info("================== PROSES TRAINING SELESAI ==================")
    
# #     return True, "Model telah berhasil dilatih ulang dan diperbarui!"


# # analisis.py
# # import os
# # import cv2
# # import numpy as np
# # import dlib
# # import face_recognition
# # import joblib
# # import logging
# # from scipy.spatial import distance as dist
# # from tensorflow.keras.models import load_model
# # from sklearn.neighbors import KNeighborsClassifier
# # import config

# # # --- State Model di Level Modul ---
# # detector = None
# # predictor = None
# # emotion_classifier = None
# # knn_clf = None

# # def load_models():
# #     """Memuat semua model yang diperlukan (Dlib, Emotion, Face Recognition) ke memori."""
# #     global detector, predictor, emotion_classifier, knn_clf
# #     logging.info("Memuat semua model...")
# #     try:
# #         detector = dlib.get_frontal_face_detector()
# #         predictor = dlib.shape_predictor(config.DLIB_SHAPE_PREDICTOR)
# #         emotion_classifier = load_model(config.EMOTION_MODEL_PATH, compile=False)
# #         if os.path.exists(config.MODEL_PATH):
# #             knn_clf = joblib.load(config.MODEL_PATH)
# #             logging.info("Semua model berhasil dimuat.")
# #         else:
# #             logging.warning("Model k-NN tidak ditemukan. Perlu training.")
# #     except Exception as e:
# #         logging.critical(f"Gagal memuat satu atau lebih model: {e}", exc_info=True)

# # def _calculate_eye_aspect_ratio(eye):
# #     """Menghitung Eye Aspect Ratio (EAR)."""
# #     A = dist.euclidean(eye[1], eye[5])
# #     B = dist.euclidean(eye[2], eye[4])
# #     C = dist.euclidean(eye[0], eye[3])
# #     return (A + B) / (2.0 * C)

# # def analyze_image_for_faces(image: np.ndarray, user_details_map: dict) -> list:
# #     """Menganalisis gambar untuk identitas, keletihan (%), dan mood (skor rincian)."""
# #     if detector is None:
# #         return []

# #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #     face_rects = detector(gray_image, 0)
# #     results = []

# #     for rect in face_rects:
# #         (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
# #         face_box = (y, x + w, y + h, x)

# #         result = {
# #             "box": (x, y, w, h), "nama": "Tidak Dikenal", "guid": None, "unit": "-",
# #             "mood": "N/A", "keletihan": 0.0,
# #             "mood_scores": {"Senang": 0.0, "Marah": 0.0, "Sedih": 0.0, "Netral": 0.0}
# #         }

# #         # 1. Pengenalan Wajah
# #         if knn_clf:
# #             encodings = face_recognition.face_encodings(rgb_image, [face_box])
# #             if encodings:
# #                 distances = knn_clf.kneighbors(encodings, n_neighbors=1)[0][0][0]
# #                 if distances <= 0.6:
# #                     label = knn_clf.predict(encodings)[0]
# #                     user_info = user_details_map.get(label)
# #                     if user_info:
# #                         result.update({
# #                             "nama": user_info.get("name"), "unit": user_info.get("unit"), "guid": user_info.get("guid")
# #                         })
        
# #         # 2. Deteksi Keletihan & Mood
# #         if predictor and emotion_classifier:
# #             shape = predictor(gray_image, rect)
# #             shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
# #             # Keletihan dalam Persentase
# #             avg_ear = (_calculate_eye_aspect_ratio(shape_np[42:48]) + _calculate_eye_aspect_ratio(shape_np[36:42])) / 2.0
# #             fatigue_level = np.interp(avg_ear, [0.18, config.FATIGUE_EAR_THRESHOLD + 0.05], [100, 0])
# #             result["keletihan"] = round(max(0, min(100, fatigue_level)), 2)

# #             # Mood dengan Skor Rincian
# #             roi_gray = gray_image[y:y + h, x:x + w]
# #             roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
# #             roi_final = np.expand_dims(np.expand_dims(roi_resized / 255.0, axis=-1), axis=0)
# #             preds = emotion_classifier.predict(roi_final, verbose=0)[0]
            
# #             result["mood"] = config.EMOTION_LABELS[preds.argmax()]
# #             all_moods = {label: float(p) * 100 for label, p in zip(config.EMOTION_LABELS, preds)}
# #             for mood_name in result["mood_scores"]:
# #                 result["mood_scores"][mood_name] = round(all_moods.get(mood_name, 0.0), 2)
        
# #         results.append(result)
# #     return results

# # def train_model() -> (bool, str):
# #     """Melatih dan menyimpan model k-NN."""
# #     global knn_clf
# #     logging.info("Memulai proses training...")
# #     encodings, labels = [], []
# #     if not os.path.isdir(config.DATASET_PATH):
# #         return False, "Direktori dataset tidak ditemukan."

# #     for user_folder in os.listdir(config.DATASET_PATH):
# #         user_path = os.path.join(config.DATASET_PATH, user_folder)
# #         if not os.path.isdir(user_path): continue
# #         for img_name in os.listdir(user_path):
# #             img_path = os.path.join(user_path, img_name)
# #             try:
# #                 image = face_recognition.load_image_file(img_path)
# #                 face_encs = face_recognition.face_encodings(image)
# #                 if face_encs:
# #                     encodings.append(face_encs[0])
# #                     labels.append(user_folder)
# #             except Exception as e:
# #                 logging.warning(f"Gagal memproses {img_path}: {e}")
    
# #     if not encodings:
# #         return False, "Training gagal, tidak ada wajah yang bisa di-encode."

# #     n_neighbors = int(round(np.sqrt(len(labels))))
# #     new_clf = KNeighborsClassifier(n_neighbors=max(1, n_neighbors), algorithm='ball_tree', weights='distance')
# #     new_clf.fit(encodings, labels)
# #     joblib.dump(new_clf, config.MODEL_PATH)
# #     knn_clf = new_clf
# #     return True, f"Training berhasil dengan {len(labels)} gambar."

# # analysis.py (Versi yang Disempurnakan)

# import os
# import cv2
# import numpy as np
# import dlib
# import face_recognition
# import joblib
# import logging
# from scipy.spatial import distance as dist
# from tensorflow.keras.models import load_model
# from sklearn.neighbors import KNeighborsClassifier

# import config

# # --- State Model di Level Modul ---
# detector = None
# predictor = None
# emotion_classifier = None
# knn_clf = None

# def load_models():
#     """Memuat semua model yang diperlukan (Dlib, Emotion, Face Recognition) ke memori."""
#     global detector, predictor, emotion_classifier, knn_clf
#     logging.info("Memuat semua model...")
#     try:
#         # Muat semua model di sini untuk efisiensi
#         detector = dlib.get_frontal_face_detector()
#         predictor = dlib.shape_predictor(config.DLIB_SHAPE_PREDICTOR)
#         emotion_classifier = load_model(config.EMOTION_MODEL_PATH, compile=False)
#         if os.path.exists(config.MODEL_PATH):
#             knn_clf = joblib.load(config.MODEL_PATH)
#         else:
#             logging.warning("Model pengenalan wajah (k-NN) tidak ditemukan. Perlu training.")
#             knn_clf = None
#         logging.info("Semua model yang tersedia berhasil dimuat.")
#     except Exception as e:
#         logging.critical(f"Gagal memuat satu atau lebih model: {e}", exc_info=True)
#         detector, predictor, emotion_classifier, knn_clf = None, None, None, None

# def _calculate_eye_aspect_ratio(eye):
#     """Menghitung Eye Aspect Ratio (EAR) untuk deteksi kantuk."""
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     # Hindari pembagian dengan nol
#     if C == 0:
#         return 0.3 # Kembalikan nilai default untuk mata terbuka jika C=0
#     return (A + B) / (2.0 * C)

# def analyze_image_for_faces(image: np.ndarray, user_details_map: dict) -> list:
#     """
#     Menganalisis gambar untuk identitas, keletihan (%), dan mood (skor rincian).
#     """
#     if detector is None or predictor is None:
#         logging.warning("Model Dlib tidak siap, analisis wajah tidak dapat dilanjutkan.")
#         return []

#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     face_rects = detector(gray_image, 0)
#     results = []

#     for rect in face_rects:
#         (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
#         face_box = (y, x + w, y + h, x)

#         # --- Penyempurnaan 1: Struktur Hasil yang Lebih Detail ---
#         result = {
#             "box": (x, y, w, h),
#             "nama": "Tidak Dikenal",
#             "guid": None,
#             "unit": "-",
#             "mood": "N/A",
#             "keletihan": 0.0,
#             "mood_scores": {
#                 "Senang": 0.0,
#                 "Marah": 0.0,
#                 "Sedih": 0.0,
#                 "Netral": 0.0
#             }
#         }

#         # --- Deteksi Orang (Pengenalan Wajah) ---
#         if knn_clf:
#             encodings = face_recognition.face_encodings(rgb_image, [face_box])
#             if encodings:
#                 distances = knn_clf.kneighbors(encodings, n_neighbors=1)[0][0][0]
#                 # Ambang batas jarak 0.45 adalah kompromi yang baik
#                 if distances <= 0.45:
#                     predicted_label = knn_clf.predict(encodings)[0]
#                     # Ambil info dari map yang disediakan oleh app.py
#                     user_info = user_details_map.get(predicted_label)
#                     if user_info:
#                         result.update({
#                             "nama": user_info.get("name"),
#                             "unit": user_info.get("unit"),
#                             "guid": user_info.get("guid")
#                         })
        
#         # --- Deteksi Keletihan & Mood ---
#         shape = predictor(gray_image, rect)
#         shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        
#         # --- Penyempurnaan 2: Keletihan dalam Format Persentase ---
#         left_eye, right_eye = shape_np[42:48], shape_np[36:42]
#         avg_ear = (_calculate_eye_aspect_ratio(left_eye) + _calculate_eye_aspect_ratio(right_eye)) / 2.0
        
#         # Petakan nilai EAR ke persentase keletihan (0-100%)
#         fatigue_level = np.interp(avg_ear, [config.EAR_THRESHOLD_CLOSED, config.EAR_THRESHOLD_OPEN], [100, 0])
#         result["keletihan"] = round(max(0, min(100, fatigue_level)), 2)

#         # --- Penyempurnaan 3: Deteksi Mood dengan Skor Rincian ---
#         if emotion_classifier:
#             roi_gray = gray_image[y:y + h, x:x + w]
#             roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#             roi_final = np.expand_dims(np.expand_dims(roi_resized / 255.0, axis=-1), axis=0)
            
#             predictions = emotion_classifier.predict(roi_final, verbose=0)[0]
            
#             # Petakan prediksi ke label emosi dari config
#             all_moods_en = {label: float(p) * 100 for label, p in zip(config.EMOTION_LABELS_ENGLISH, predictions)}
            
#             # Dapatkan mood dominan dan terjemahkan
#             dominant_mood_en = max(all_moods_en, key=all_moods_en.get)
#             result["mood"] = config.EMOTION_MAP_TO_INDONESIAN[dominant_mood_en]

#             # Isi `mood_scores` dengan nama field dalam Bahasa Indonesia
#             result["mood_scores"]["Senang"] = round(all_moods_en.get("happy", 0.0), 2)
#             result["mood_scores"]["Marah"] = round(all_moods_en.get("angry", 0.0), 2)
#             result["mood_scores"]["Sedih"] = round(all_moods_en.get("sad", 0.0), 2)
#             result["mood_scores"]["Netral"] = round(all_moods_en.get("neutral", 0.0), 2)
        
#         results.append(result)

#     return results

# def train_model() -> (bool, str):
#     """
#     Melatih model k-NN dari dataset dan menyimpannya ke file.
#     """
#     global knn_clf
#     logging.info("Memulai proses training model k-NN...")
#     encodings, labels = [], []
    
#     if not os.path.isdir(config.DATASET_PATH):
#         msg = "Direktori dataset tidak ditemukan."
#         logging.error(msg)
#         return False, msg

#     # Iterasi folder dan gambar untuk mengumpulkan data training
#     for user_folder in os.listdir(config.DATASET_PATH):
#         user_path = os.path.join(config.DATASET_PATH, user_folder)
#         if not os.path.isdir(user_path): continue
#         for img_name in os.listdir(user_path):
#             img_path = os.path.join(user_path, img_name)
#             try:
#                 # Gunakan face_recognition untuk memuat gambar agar formatnya konsisten
#                 image = face_recognition.load_image_file(img_path)
#                 face_encs = face_recognition.face_encodings(image)
#                 if face_encs:
#                     encodings.append(face_encs[0])
#                     # --- UBAH BAGIAN INI ---
#                     # Ambil HANYA GUID dari nama folder sebagai label
#                     try:
#                         guid_label = user_folder.split('_')[-1]
#                         labels.append(guid_label)
#                     except IndexError:
#                         logging.warning(f"Format folder {user_folder} salah, tidak bisa mendapatkan GUID. Dilewati.")
#                     # -------------------------
#                     # Labelnya adalah nama folder, misal: "Asep_Trisna_Setiawan_guid-123"
#                     # labels.append(user_folder)
#             except Exception as e:
#                 logging.warning(f"Gagal memproses gambar {img_path}: {e}")
    
#     if not encodings:
#         msg = "Training gagal, tidak ada wajah yang berhasil di-encode dari dataset."
#         logging.error(msg)
#         return False, msg

#     # --- Penyempurnaan 4: Logika Training yang Lebih Rapi ---
#     n_neighbors = max(1, int(round(np.sqrt(len(labels)))))
#     new_clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
#     new_clf.fit(encodings, labels)
    
#     # Simpan model yang telah dilatih
#     joblib.dump(new_clf, config.MODEL_PATH)
    
#     # Perbarui model di memori agar langsung aktif
#     knn_clf = new_clf
    
#     msg = f"Training berhasil dengan {len(labels)} gambar dari {len(np.unique(labels))} orang."
#     logging.info(msg)
#     return True, msg

# analysis.py (Versi Final)

# import os
# import cv2
# import numpy as np
# import dlib
# import face_recognition
# import joblib
# import logging
# from scipy.spatial import distance as dist
# from tensorflow.keras.models import load_model
# from sklearn.neighbors import KNeighborsClassifier
# from tensorflow.keras.preprocessing.image import img_to_array
# import config

# class FaceAnalyzer:
#     def __init__(self):
#         self.detector = dlib.get_frontal_face_detector()
#         self.predictor = None
#         self.emotion_classifier = None
#         self.knn_clf = None

#     def load_models(self):
#         logging.info("Memuat semua model AI...")
#         try:
#             self.predictor = dlib.shape_predictor(config.DLIB_SHAPE_PREDICTOR)
#             self.emotion_classifier = load_model(config.EMOTION_MODEL_PATH, compile=False)
#             if os.path.exists(config.MODEL_PATH):
#                 self.knn_clf = joblib.load(config.MODEL_PATH)
#                 logging.info(f"Model k-NN berhasil dimuat dari: {config.MODEL_PATH}")
#             else:
#                 logging.warning("Model k-NN tidak ditemukan. Perlu training.")
#         except Exception as e:
#             logging.critical(f"Gagal memuat model: {e}", exc_info=True)
#             raise

#     def _calculate_ear(self, eye):
#         A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4]); C = dist.euclidean(eye[0], eye[3])
#         return (A + B) / (2.0 * C) if C != 0 else 0.3

#     def recognize_fast(self, image: np.ndarray, user_details_map: dict):
#         """Fungsi analisis CEPAT untuk Flask: hanya identifikasi."""
#         if not self.knn_clf or not self.detector: return None
        
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         face_rects = self.detector(rgb_image, 0)
        
#         if not face_rects: return None

#         rect = face_rects[0]
#         (x,y,w,h) = (rect.left(), rect.top(), rect.width(), rect.height())
#         face_box = (y, x+w, y+h, x)
        
#         encodings = face_recognition.face_encodings(rgb_image, [face_box])
#         if encodings:
#             distances = self.knn_clf.kneighbors(encodings, n_neighbors=1)[0][0][0]
#             if distances <= 0.45:
#                 predicted_guid = self.knn_clf.predict(encodings)[0]
#                 user_info = user_details_map.get(predicted_guid, {})
#                 return {
#                     "guid": predicted_guid,
#                     "name": user_info.get("name", "Tidak Dikenal"),
#                     "unit": user_info.get("unit", "-"),
#                     "box_coords": (x,y,w,h)
#                 }
#         return None

#     def analyze_deep(self, image: np.ndarray, user_info: dict):
#         """Fungsi analisis LENGKAP untuk Consumer: HANYA mood & keletihan."""
#         if not all([self.detector, self.predictor, self.emotion_classifier]):
#             logging.error("Model tidak siap untuk analisis mendalam.")
#             return None, None

#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         face_rects = self.detector(gray, 0)
#         if not face_rects: return None, None
        
#         rect = face_rects[0]
#         (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        
#         # --- PERUBAHAN UTAMA: Langsung gunakan user_info dari payload ---
#         result = user_info.copy()

#         # --- Lakukan HANYA analisis keletihan & mood ---
#         shape = self.predictor(gray, rect)
#         landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
#         avg_ear = (self._calculate_ear(landmarks[42:48]) + self._calculate_ear(landmarks[36:42])) / 2.0
#         fatigue = np.interp(avg_ear, [config.EAR_THRESHOLD_CLOSED, config.EAR_THRESHOLD_OPEN], [100, 0])
#         result["keletihan"] = round(max(0, min(100, fatigue)), 2)

#         roi = gray[y:y+h, x:x+w]
#         roi = cv2.resize(roi, (48, 48))
#         roi = np.expand_dims(np.expand_dims(roi.astype("float") / 255.0, axis=-1), axis=0)
#         preds = self.emotion_classifier.predict(roi, verbose=0)[0]
#         moods_en = {label: p*100 for label, p in zip(config.EMOTION_LABELS_ENGLISH, preds)}
#         dom_mood = max(moods_en, key=moods_en.get)
#         result["mood"] = config.EMOTION_MAP_TO_INDONESIAN[dom_mood]
#         result.update({
#             "Marah": round(moods_en.get('angry', 0.0), 2),
#             "Senang": round(moods_en.get('happy', 0.0), 2),
#             "Sedih": round(moods_en.get('sad', 0.0), 2),
#             "Netral": round(moods_en.get('neutral', 0.0), 2),
#         })
#         result["status_absen"] = "Terdeteksi"
#         # ----------------------------------------------------------------

#         return result, (y, x+w, y+h, x)

#     def train_model(self):
#         logging.info("Memulai proses training model k-NN...")
#         encodings, labels = [], []
#         if not os.path.isdir(config.DATASET_PATH): return False, "Direktori dataset tidak ditemukan."

#         for user_folder in os.listdir(config.DATASET_PATH):
#             user_path = os.path.join(config.DATASET_PATH, user_folder)
#             if not os.path.isdir(user_path): continue
#             for img_name in os.listdir(user_path):
#                 img_path = os.path.join(user_path, img_name)
#                 try:
#                     image = face_recognition.load_image_file(img_path)
#                     face_encs = face_recognition.face_encodings(image)
#                     if face_encs:
#                         encodings.append(face_encs[0])
#                         guid_label = user_folder.split('_')[-1]
#                         labels.append(guid_label)
#                 except Exception as e:
#                     logging.warning(f"Gagal memproses {img_path}: {e}")
        
#         if not encodings: return False, "Training gagal, tidak ada wajah bisa di-encode."

#         n_neighbors = max(1, int(round(np.sqrt(len(labels)))))
#         knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
#         knn_clf.fit(encodings, labels)
        
#         joblib.dump(knn_clf, config.MODEL_PATH)
#         self.knn_clf = knn_clf
#         msg = f"Training berhasil dengan {len(labels)} gambar dari {len(np.unique(labels))} orang."
#         logging.info(msg)
#         return True, msg

#     def draw_labels_on_image(self, image, face_location, analysis_results):
#         top, right, bottom, left = face_location
#         label_text = f"{analysis_results.get('name', 'N/A')} | {analysis_results.get('mood', 'N/A')} | Keletihan: {analysis_results.get('keletihan', 0):.1f}%"
#         color = (0, 255, 0)
#         if analysis_results.get('mood') == 'Marah': color = (0, 0, 255)
#         elif analysis_results.get('mood') == 'Sedih': color = (255, 0, 0)
#         cv2.rectangle(image, (left, top), (right, bottom), color, 2)
#         cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
#         cv2.putText(image, label_text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
#         return image