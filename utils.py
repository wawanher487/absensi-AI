import logging
import requests
import config

def setup_logging():
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_and_map_users_from_api():
    """
    Fetches user data from the configured API and maps it by a unique identifier.
    The unique identifier is 'name_guid' for easy lookup after face recognition.
    """
    logging.info("Fetching and mapping user details from API...")
    user_details_map = {}
    try:
        # Set a timeout for the request to prevent indefinite hanging.
        response = requests.get(config.API_URL, timeout=15)
        # Raise an exception for bad status codes (4xx or 5xx).
        response.raise_for_status()
        users = response.json().get('data', [])

        for user in users:
            # Create a unique key for each user.
            user_key = f"{user['name'].replace(' ', '_')}_{user['guid']}"
            user_details_map[user_key] = user

        logging.info(f"Successfully fetched and mapped {len(user_details_map)} users.")
        return user_details_map

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch user data from API: {e}")
        # Return an empty map if the API call fails, so the app can still run.
        return {}
