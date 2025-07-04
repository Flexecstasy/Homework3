import os
import time
import logging

def setup_logging(log_file: str):
    # Создаём директорию для логов, если не существует
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
        logging.info(f"Elapsed time: {self.elapsed:.4f} sec")