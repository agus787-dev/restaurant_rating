import logging
from pathlib import Path


class Logger:

    def __init__(self, name, log_dir="logs"):

        # PROJECT ROOT
        ROOT_DIR = Path(__file__).resolve().parent.parent

        # LOG DIRECTORY
        LOG_DIR = ROOT_DIR / log_dir

        # CREATE LOGS FOLDER IF NOT EXISTS
        LOG_DIR.mkdir(exist_ok=True)

        # LOG FILE PATH
        log_file = LOG_DIR / f"{name}.log"

        # LOGGER OBJECT
        self.logger = logging.getLogger(name)

        # AVOID DUPLICATE HANDLERS
        if not self.logger.handlers:

            self.logger.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )

            file_handler = logging.FileHandler(log_file)

            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)