import os
import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger(log_file):
    """
    Sets up a logger with daily rotation and 10-day retention.

    Args:
        log_file (str): The full path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is already configured
    if not logger.handlers:
        # Ensure the directory for logs exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create TimedRotatingFileHandler
        handler = logging.FileHandler(log_file, encoding='utf-8')

        # Create Formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

        # Also add StreamHandler to output logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.info("Logging system initialized.")

    return logger
