import os
import cv2
import numpy as np
import base64
import time
import json
import logging
import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import BadRequest
import threading

# --- Local Module Imports ---
import config
from utils import setup_logging, get_and_map_users_from_api
from services import ftp_service, rmq_service
from analysis import face_analyzer

# --- Initial Setup ---
setup_logging()
logging.info("Flask application starting...")
app = Flask(__name__)

# --- Load User Data and Models ---
user_details_map = get_and_map_users_from_api()
face_analyzer.load_models()

# --- State Management ---
last_detection_timestamps = {}
training_lock = threading.Lock()

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main page with a list of users."""
    global user_details_map # Diperlukan untuk memodifikasi variabel global
    logging.info(f"Request for main page from {request.remote_addr}")

    # [PERBAIKAN] Jika daftar pengguna kosong (misal, gagal saat startup), coba ambil lagi.
    if not user_details_map:
        logging.warning("User map is empty. Attempting to re-fetch from API...")
        user_details_map = get_and_map_users_from_api()

    users = list(user_details_map.values())
    if not users:
        logging.warning("Still no users found after re-fetch attempt. Rendering with an empty list.")
    
    return render_template('index.html', users=users)

@app.route('/capture', methods=['POST'])
def capture():
    """
    Handles the capturing and saving of new face images for training.
    Expects a JSON payload with 'guid', 'name', and 'image' data.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'name' not in data:
            raise BadRequest("Missing required fields in request.")

        user_guid = data.get('guid', 'unknown')
        user_name = data['name']
        image_data_b64 = data['image'].split(',')[1]

        user_folder = os.path.join(config.DATASET_PATH, f"{user_name.replace(' ', '_')}_{user_guid}")
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            logging.info(f"Created new dataset folder: {user_folder}")

        img_bytes = base64.b64decode(image_data_b64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'status': 'error', 'message': 'Invalid image data received.'}), 400

        image_path = os.path.join(user_folder, f"image_{len(os.listdir(user_folder)) + 1}.jpg")
        cv2.imwrite(image_path, img)
        logging.info(f"Successfully saved new image to {image_path}")

        return jsonify({'status': 'success', 'message': f'Gambar untuk {user_name} telah disimpan!'})

    except (BadRequest, KeyError) as e:
        logging.error(f"Bad request in /capture: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logging.error(f"Error processing captured image: {e}")
        return jsonify({'status': 'error', 'message': 'Gagal menyimpan gambar.'}), 500


@app.route('/train', methods=['GET'])
def train_model():
    """
    Initiates the training process synchronously.
    Handles concurrent requests by locking the process.
    """
    if not training_lock.acquire(blocking=False):
        logging.warning("Training request received while another training is in progress.")
        return jsonify({
            'status': 'error',
            'message': 'Proses training sedang berjalan. Silakan coba beberapa saat lagi.'
        }), 409

    logging.info("Training process initiated by user. Lock acquired.")
    try:
        success, message = face_analyzer.train_model()
        if success:
            logging.info("Training completed and model reloaded successfully!")
            return jsonify({'status': 'success', 'message': message})
        else:
            logging.warning(f"Training failed: {message}")
            return jsonify({'status': 'error', 'message': message}), 400
    except Exception as e:
        logging.critical(f"An unexpected error occurred during training: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Terjadi kesalahan internal saat training.'}), 500
    finally:
        training_lock.release()
        logging.info("Training process finished. Lock released.")


@app.route('/recognize_frame', methods=['POST'])
def recognize_frame():
    """
    Receives a video frame, analyzes it for faces, and processes presence if a known user is detected.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify([])

        image_data_b64 = data['image'].split(',')[1]
        latitude = data.get('latitude', 0.0)
        longitude = data.get('longitude', 0.0)

        img_bytes = base64.b64decode(image_data_b64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            logging.warning("Received an invalid/empty image frame for recognition.")
            return jsonify([])

    except (IndexError, base64.binascii.Error) as e:
        logging.error(f"Could not decode base64 image from request: {e}")
        return jsonify([])
    except Exception as e:
        logging.error(f"Unexpected error processing request for /recognize_frame: {e}")
        return jsonify([])

    results = face_analyzer.analyze_image_for_faces(img, user_details_map)

    if results and results[0].get('guid'):
        person = results[0]
        user_guid = person['guid']
        current_time = time.time()
        cooldown_period = config.DETECTION_COOLDOWN_SECONDS

        if (current_time - last_detection_timestamps.get(user_guid, 0)) > cooldown_period:
            logging.info(f"User '{person.get('name')}' detected. Cooldown passed. Processing presence...")
            last_detection_timestamps[user_guid] = current_time

            _, buffer = cv2.imencode('.jpg', img)
            image_bytes_for_upload = buffer.tobytes()

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            remote_filename = f"detection_{user_guid}_{timestamp}.jpg"

            logging.info(f"Attempting to upload '{remote_filename}' to FTP...")
            if ftp_service.upload_to_ftp(image_bytes_for_upload, remote_filename):
                logging.info(f"FTP STATUS: SUCCESS uploading '{remote_filename}'.")

                # --- Kirim ke RMQ #1 (Presensi) ---
                image_url = f"{config.FTP_BASE_URL}{config.FTP_FOLDER}/{remote_filename}"
                presence_payload = rmq_service.create_presence_payload(
                    user_guid=user_guid,
                    user_name=person.get('name', 'N/A'),
                    image_url=image_url,
                    latitude=latitude,
                    longitude=longitude
                )
                
                logging.info(f"Attempting to publish presence message to RMQ #1...")
                if rmq_service.publish_to_rmq(presence_payload):
                    logging.info("RMQ #1 STATUS: SUCCESS sending presence message.")
                    results[0]['presence_sent'] = True
                else:
                    logging.error("RMQ #1 STATUS: FAILED to send presence message.")
                
                # --- Kirim ke RMQ #2 (Notifikasi File) ---
                notification_payload = rmq_service.create_file_notification_payload(
                    filename=remote_filename
                )

                logging.info(f"Attempting to publish file notification to RMQ #2...")
                if rmq_service.publish_file_notification(notification_payload):
                    logging.info("RMQ #2 STATUS: SUCCESS sending file notification.")
                else:
                    logging.error("RMQ #2 STATUS: FAILED to send file notification.")
            else:
                logging.error(f"FTP STATUS: FAILED to upload '{remote_filename}'. RMQ messages will not be sent.")
        else:
            logging.info(f"User '{person.get('name')}' detected. Cooldown active. Skipping presence processing.")

    return jsonify(results)


@app.route('/get_training_stats')
def get_training_stats():
    """Endpoint to get statistics about the training dataset, such as image count."""
    logging.info("Request received for training stats.")
    try:
        image_count = 0
        if os.path.exists(config.DATASET_PATH):
            for user_folder in os.listdir(config.DATASET_PATH):
                user_path = os.path.join(config.DATASET_PATH, user_folder)
                if os.path.isdir(user_path):
                    image_count += len([
                        name for name in os.listdir(user_path)
                        if name.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ])
        logging.info(f"Calculated image count: {image_count}")
        return jsonify({'image_count': image_count})
    except Exception as e:
        logging.error(f"Error calculating training stats: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.APP_PORT, debug=False, threaded=True)
