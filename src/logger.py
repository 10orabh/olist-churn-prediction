import logging
import os 

def get_logger(name):
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
    
        log_file_path = os.path.join(log_dir, "app.log")

    
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger