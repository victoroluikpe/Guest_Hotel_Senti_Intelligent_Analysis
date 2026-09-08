import logging
import os
from datetime import datetime

LOG_DIRS = "logs"
os.makedirs(LOG_DIRS, exist_ok=True)

def configure_logger():
    logger = logging.getLogger()

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s %(name)s - %(levelname)s - %(message)s]"
    ) 

    # creating a new file for every run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file = os.path.join(LOG_DIRS, f"run_{timestamp}.log")

    # filehandler it handle the streaming process of log in a specific file
    file_handler = logging.FileHandler(
        log_file, encoding="utf-8"

    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
