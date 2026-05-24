import numpy as np
import pandas as pd
from src.logger import Logger


class FeatureTransformer:
    def __init__(self, skew_threshold: float = 0.5):
        self.skew_threshold = skew_threshold
        self.log_transformed_cols = []
        self.log = Logger("feature_transformer")

    def log_transform_train(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.info("Log transform started")
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            skewness = df[col].skew()
            if abs(skewness) > self.skew_threshold:
                if (df[col] >= 0).all():
                    df[col] = np.log1p(df[col])

                self.log_transformed_cols.append(col)
                self.log.info(f"  Log transform: {col} (skew: {skewness:.2f})")

        self.log.info(f"Log transform finished: {len(self.log_transformed_cols)} columns")
        return df

    def log_transform_test(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.info("Log transform (test) started")
        df = df.copy()

        for col in self.log_transformed_cols:
            if col in df.columns:
                 if (df[col] >= 0).all():
                    df[col] = np.log1p(df[col])

        self.log.info("Log transform (test) finished")
        return df

    def apply_log_transform(self, x_train: pd.DataFrame, x_test: pd.DataFrame):
        x_train_transformed = self.log_transform_train(x_train)
        x_test_transformed = self.log_transform_test(x_test)
        return x_train_transformed, x_test_transformed