import logging
import json
import pika
import datetime
import config

def publish_to_rmq(payload: dict) -> bool:
    """
    Publishes a message payload to the configured RabbitMQ queue.

    Args:
        payload: A dictionary representing the message to be sent.

    Returns:
        True if publishing was successful, False otherwise.
    """
    connection = None
    try:
        params = pika.URLParameters(config.RMQ_URI)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()

        # [PERUBAIKAN] Menyesuaikan deklarasi queue dengan konfigurasi server.
        # durable diubah menjadi False untuk mengatasi error 'incompatible arguments'.
        # Jika Anda ingin queue ini durable (pesan tidak hilang saat server restart),
        # Anda harus menghapus queue 'presence-2024' dari RabbitMQ Management UI
        # lalu mengubah durable menjadi True di sini.
        #
        # Menambahkan TTL (Time-To-Live) 1 menit untuk pesan.
        queue_args = {
            "x-message-ttl": 60000
        }
        channel.queue_declare(
            queue=config.RMQ_QUEUE, 
            durable=False, # Diubah dari True menjadi False agar sesuai dengan server
            arguments=queue_args
        )

        message_body = json.dumps(payload)

        # Publish the message with a persistent delivery mode.
        channel.basic_publish(
            exchange='',
            routing_key=config.RMQ_QUEUE,
            body=message_body,
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
        )
        logging.info(f"Successfully published message to RMQ queue '{config.RMQ_QUEUE}'.")
        return True
    except Exception as e:
        logging.error(f"RMQ PUBLISH FAILED: {e}", exc_info=True)
        return False
    finally:
        # Ensure the connection is closed.
        if connection and connection.is_open:
            connection.close()

def create_presence_payload(user_guid: str, user_name: str, image_url: str, latitude: float, longitude: float) -> dict:
    """
    Creates a standardized payload for a presence report.
    """
    return {
        "pattern": "daily-report",
        "data": {
            "id": user_guid,
            "type": "AI_Detection",
            "presenceType": "Daily Presence",
            "description": f"Face detection successful: {user_name}",
            "imageUrl": image_url,
            "latitude": latitude,
            "longitude": longitude,
            # "guidInstitution": config.GUID_INSTITUTION,
            "hour": datetime.datetime.now().strftime("%H.%M")
        }
    }


# # --- [FUNGSI BARU] ---

def publish_file_notification(payload: dict) -> bool:
    """
    Publishes a file notification payload to the second RabbitMQ queue.
    """
    connection = None
    try:
        print("RMQ_URI:", config.RMQ2_URI)  # Debugging line to check RMQ2 URI
        params = pika.URLParameters(config.RMQ2_URI)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()

        # Declare the queue. Assuming it should be durable for notifications.
        # channel.queue_declare(queue=config.RMQ2_QUEUE, durable=False)
        queue_args = {
            "x-message-ttl": 60000
        }
        channel.queue_declare(
            queue=config.RMQ2_QUEUE, 
            durable=False, # Diubah dari True menjadi False agar sesuai dengan server
            arguments=queue_args
        )

        message_body = json.dumps(payload)

        # message_body = json.dumps(payload)

        channel.basic_publish(
            exchange='',
            routing_key=config.RMQ2_QUEUE,
            body=message_body,
            # properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
        )
        logging.info(f"Successfully published file notification to RMQ queue '{config.RMQ2_QUEUE}'.")
        return True
    except Exception as e:
        logging.error(f"RMQ (Notification) PUBLISH FAILED: {e}", exc_info=True)
        return False
    finally:
        if connection and connection.is_open:
            connection.close()

def create_file_notification_payload(filename: str) -> dict:
    """
    Creates a standardized payload for a file notification report.
    """
    return {
        "filename": filename,
        "guid_camera": config.CAMERA_GUID,
        "capture_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }