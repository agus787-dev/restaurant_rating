import pandas as pd
import os
import klib
from sklearn.model_selection import train_test_split
from src.logger import Logger

log = Logger("data_loader")

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.df   = None

    def load(self):
        self.df = pd.read_csv(self.path)
        self.df = klib.clean_column_names(self.df)
        log.info(f"Data yuklandi: {self.df.shape[0]} qator, {self.df.shape[1]} ustun")
        return self.df

    def split(self, target_column, test_size=0.2, to_drop=[]):
        X = self.df.drop(columns=[target_column] + to_drop)
        y = self.df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        log.info(f"Train va testga split qilindi => Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test