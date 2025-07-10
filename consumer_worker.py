import pika
import json
from services.db_service import save_detection_history
from services.ftp_service import get_ftp_image_url
import config

def callback(ch, method, properties, body):
    print("[WORKER-AI] Menerima pesan...")
    data = json.loads(body)
    
    image_path = data['gambar']  # misalnya: 2025-07-04/abc.jpg
    image_url = get_ftp_image_url(image_path)  # fungsi bantu buat URL penuh
    
    person_data = {
        "name": "Unknown",  # akan diisi dari face recognition nanti
        "mood": "Netral",
        "keletihan": 12.0,
        "mood_scores": {
            "Marah": 0,
            "Senang": 0,
            "Netral": 100,
            "Sedih": 0
        },
        "guid": data["guid"],  # bisa juga dari hasil klasifikasi
        "unit": "IT"
    }

    # Panggil fungsi klasifikasi wajah kamu di sini (face_recognize.py / classifier.py)
    # dan hasilnya dimasukkan ke person_data

    save_detection_history(person_data, image_url)

def main():
    connection = pika.BlockingConnection(pika.URLParameters(config.RMQ_URI))
    channel = connection.channel()
    channel.queue_declare(queue='to_ai_service', durable=False)
    channel.basic_consume(queue='to_ai_service', on_message_callback=callback, auto_ack=True)

    print("[WORKER-AI] Menunggu pesan dari queue 'to_ai_service'...")
    channel.start_consuming()

if __name__ == "__main__":
    main()
