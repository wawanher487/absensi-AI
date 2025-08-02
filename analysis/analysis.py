# Di dalam file: analysis/analysis.py

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
from tensorflow.keras.preprocessing.image import img_to_array
import config

class FaceAnalyzer:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.emotion_classifier = None
        self.knn_clf = None

    def load_models(self):
        logging.info("Memuat semua model AI...")
        try:
            self.predictor = dlib.shape_predictor(config.DLIB_SHAPE_PREDICTOR)
            self.emotion_classifier = load_model(config.EMOTION_MODEL_PATH, compile=False)
            if os.path.exists(config.MODEL_PATH):
                self.knn_clf = joblib.load(config.MODEL_PATH)
                logging.info(f"Model k-NN berhasil dimuat dari: {config.MODEL_PATH}")
            else:
                logging.warning("Model k-NN tidak ditemukan. Perlu training.")
        except Exception as e:
            logging.critical(f"Gagal memuat model: {e}", exc_info=True)
            raise

    def _calculate_ear(self, eye):
        A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4]); C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.3

    def analyze_image_for_faces(self, image: np.ndarray, user_details_map: dict) -> list:
        if self.detector is None or self.predictor is None:
            logging.warning("Model Dlib tidak siap, analisis wajah tidak dapat dilanjutkan.")
            return []

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_rects = self.detector(gray_image, 0)
        results = []

        for rect in face_rects:
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            face_box = (y, x + w, y + h, x)
            result = {"box": (x,y,w,h), "nama": "Tidak Dikenal", "guid": None, "unit": "-", "mood": "N/A", "keletihan": 0.0, "mood_scores": {}}
            
            if self.knn_clf:
                encodings = face_recognition.face_encodings(rgb_image, [face_box])
                if encodings:
                    distances = self.knn_clf.kneighbors(encodings, n_neighbors=1)[0][0][0]
                    if distances <= 0.45:
                        predicted_label = self.knn_clf.predict(encodings)[0]
                        user_info = user_details_map.get(predicted_label)
                        if user_info:
                            result.update({"nama": user_info.get("name"), "unit": user_info.get("unit"), "guid": user_info.get("guid")})

            shape = self.predictor(gray_image, rect)
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            avg_ear = (self._calculate_ear(landmarks[42:48]) + self._calculate_ear(landmarks[36:42])) / 2.0
            fatigue_level = np.interp(avg_ear, [config.EAR_THRESHOLD_CLOSED, config.EAR_THRESHOLD_OPEN], [100, 0])
            result["keletihan"] = round(max(0, min(100, fatigue_level)), 2)
            
            if self.emotion_classifier:
                roi_gray = gray_image[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi_final = np.expand_dims(np.expand_dims(roi_resized / 255.0, axis=-1), axis=0)
                predictions = self.emotion_classifier.predict(roi_final, verbose=0)[0]
                all_moods_en = {label: float(p) * 100 for label, p in zip(config.EMOTION_LABELS_ENGLISH, predictions)}
                dominant_mood_en = max(all_moods_en, key=all_moods_en.get)
                result["mood"] = config.EMOTION_MAP_TO_INDONESIAN.get(dominant_mood_en, "N/A")
                result["mood_scores"] = { "Senang": round(all_moods_en.get("happy", 0.0), 2), "Marah": round(all_moods_en.get("angry", 0.0), 2), "Sedih": round(all_moods_en.get("sad", 0.0), 2), "Netral": round(all_moods_en.get("neutral", 0.0), 2) }
            
            results.append(result)

        return results

    def train_model(self) -> (bool, str):
        logging.info("Memulai proses training model k-NN...")
        encodings, labels = [], []
        if not os.path.isdir(config.DATASET_PATH): return False, "Direktori dataset tidak ditemukan."

        for user_folder in os.listdir(config.DATASET_PATH):
            user_path = os.path.join(config.DATASET_PATH, user_folder)
            if not os.path.isdir(user_path): continue
            for img_name in os.listdir(user_path):
                img_path = os.path.join(user_path, img_name)
                try:
                    image = face_recognition.load_image_file(img_path)
                    face_encs = face_recognition.face_encodings(image)
                    if face_encs:
                        encodings.append(face_encs[0])
                        guid_label = user_folder.split('_')[-1]
                        labels.append(guid_label)
                except Exception as e:
                    logging.warning(f"Gagal memproses gambar {img_path}: {e}")
        
        if not encodings: return False, "Training gagal, tidak ada wajah bisa di-encode."
        
        n_neighbors = max(1, int(round(np.sqrt(len(labels)))))
        new_clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
        new_clf.fit(encodings, labels)
        
        joblib.dump(new_clf, config.MODEL_PATH)
        self.knn_clf = new_clf
        msg = f"Training berhasil dengan {len(labels)} gambar dari {len(np.unique(labels))} orang."
        logging.info(msg)
        return True, msg