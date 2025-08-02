import logging
from ftplib import FTP
from io import BytesIO
import config

def upload_to_ftp(image_bytes: bytes, remote_filename: str) -> bool:
    """
    Uploads a byte stream representation of an image to the FTP server.

    Args:
        image_bytes: The image data as a byte string.
        remote_filename: The desired filename on the FTP server.

    Returns:
        True if the upload was successful, False otherwise.
    """
    try:
        # Using 'with' ensures the connection is properly closed even if errors occur.
        with FTP() as ftp:
            # Set a timeout for the connection attempt.
            logging.info("Connecting to FTP server...")
            ftp.connect(config.FTP_HOST, config.FTP_PORT, timeout=30)
            logging.info("Connected. Logging in...")
            ftp.login(config.FTP_USER, config.FTP_PASS)
            logging.info("Logged in successfully.")
            ftp.set_pasv(True) 

            
            # [DIUBAH] More robust way to handle directory existence.
            try:
                # Attempt to change into the target directory.
                ftp.cwd(config.FTP_FOLDER)
                logging.info(f"Successfully changed to FTP directory: '{config.FTP_FOLDER}'")
            except Exception as e:
                # If changing directory fails, it probably doesn't exist.
                logging.warning(f"Could not change to directory '{config.FTP_FOLDER}'. Attempting to create it. Error: {e}")
                try:
                    # Create the directory.
                    ftp.mkd(config.FTP_FOLDER)
                    # Attempt to change into it again after creation.
                    ftp.cwd(config.FTP_FOLDER)
                    logging.info(f"Successfully created and changed to FTP directory: '{config.FTP_FOLDER}'")
                except Exception as create_error:
                    logging.error(f"FATAL: Failed to create or access FTP directory '{config.FTP_FOLDER}'. Error: {create_error}")
                    return False # Cannot proceed if directory is inaccessible.

            # Use BytesIO to treat the byte string as a file-like object.
            with BytesIO(image_bytes) as file_obj:
                ftp.storbinary(f'STOR {remote_filename}', file_obj)

            logging.info(f"Successfully uploaded {remote_filename} to FTP.")
            return True
    except Exception as e:
        # Catching a broad exception, but logging the specific error is key.
        logging.error(f"FTP UPLOAD FAILED for '{remote_filename}': {e}", exc_info=True)
        return False
    

def get_ftp_image_url(filename: str) -> str:
    return f"https://monja-file.pptik.id/v1/view?path={filename}"

