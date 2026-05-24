import os
import joblib
import pandas as pd
from src.logger import Logger


class DataSaver:
    def __init__(self, base_path: str, version: str = "baseline"):
        self.base_path = base_path
        self.version = version
        self.data_dir = os.path.join(base_path, "data", "preprocessed", version)
        self.models_dir = os.path.join(base_path, "models", version)
        self.logger = Logger("data_saver")

    def save_files(self, **dataframes):
        os.makedirs(self.data_dir, exist_ok=True)

        for name, data in dataframes.items():
            full_path = os.path.join(self.data_dir, f"{name}.csv")

            try:
                data.to_csv(full_path, index=False)
                self.logger.info(f"Saved: {full_path}")
            except Exception as e:
                self.logger.error(f"{name} did not saved: {e}")
                raise

    def save_models(self, **models):
        os.makedirs(self.models_dir, exist_ok=True)

        for name, model in models.items():
            full_path = os.path.join(self.models_dir, f"{name}.joblib")

            try:
                joblib.dump(model, full_path)
                self.logger.info(f"Saved: {full_path}")

            except Exception as e:
                self.logger.error(f"{name} did not saved: {e}")
                raise