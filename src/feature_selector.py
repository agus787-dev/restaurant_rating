import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from src.logger import Logger


class FeatureSelector:
    def __init__(self):
        self.correlation_cols_to_drop = []
        self.importance_cols = []
        self.rfe_cols = []
        self.log = Logger("feature_selector")

    # 1. CORRELATION
    def correlation_selection(self, df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        self.log.info(f"Correlation selection started (threshold={threshold})")
        df = df.copy()

        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.correlation_cols_to_drop = [
            col for col in upper.columns if any(upper[col] > threshold)
        ]

        df = df.drop(columns=self.correlation_cols_to_drop)

        self.log.info(f"Correlation selection finished: {len(self.correlation_cols_to_drop)} columns removed => {self.correlation_cols_to_drop}")
        print(f"columns removed: ", self.correlation_cols_to_drop)
        return df

    def apply_correlation_selection(self, x_train: pd.DataFrame, x_test: pd.DataFrame, threshold: float = 0.9):
        x_train_selected = self.correlation_selection(x_train, threshold=threshold)
        x_test_selected = x_test.drop(columns=self.correlation_cols_to_drop, errors="ignore")
        return x_train_selected, x_test_selected

    # 2. FEATURE IMPORTANCE
    def importance_selection(self, x_train: pd.DataFrame, y_train: pd.Series, threshold: float = 0.01) -> pd.DataFrame:
        self.log.info(f"Importance selection started (threshold={threshold})")

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)

        importance_df = pd.DataFrame({
            "feature": x_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        print(importance_df)

        self.importance_cols = importance_df[
            importance_df["importance"] >= threshold
        ]["feature"].tolist()

        self.log.info(f"Importance selection finished: {len(self.importance_cols)} columns saved")
        print("Columns: ", self.importance_cols)
        return importance_df

    def apply_importance_selection(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, threshold: float = 0.01):
        self.importance_selection(x_train, y_train, threshold=threshold)
        x_train_selected = x_train[self.importance_cols]
        x_test_selected = x_test[self.importance_cols]
        return x_train_selected, x_test_selected

    # 3. RFE
    def rfe_selection(self, x_train: pd.DataFrame, y_train: pd.Series, n_features: int = 10) -> pd.DataFrame:
        self.log.info(f"RFE selection started (n_features={n_features})")

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        rfe.fit(x_train, y_train)

        self.rfe_cols = x_train.columns[rfe.support_].tolist()

        rfe_df = pd.DataFrame({
            "feature": x_train.columns,
            "selected": rfe.support_,
            "ranking": rfe.ranking_
        }).sort_values("ranking")

        self.log.info(f"RFE selection finished: {self.rfe_cols}")
        return rfe_df

    def apply_rfe_selection(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, n_features: int = 10):
        self.rfe_selection(x_train, y_train, n_features=n_features)
        x_train_selected = x_train[self.rfe_cols]
        x_test_selected = x_test[self.rfe_cols]
        return x_train_selected, x_test_selected