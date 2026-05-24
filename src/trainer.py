import joblib
import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger import Logger


class Trainer:
    def __init__(self, version: str = "baseline"):
        self.version = version
        self.models_dir = os.path.join("..", "models", version)
        self.results_dir = os.path.join("..", "results", version)
        self.model = None
        self.model_name = None
        self.log = Logger("trainer")

    def _get_model(self, model_name: str):
        models = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        }
        if model_name not in models:
            raise ValueError(f"Invalid model: {model_name}. Exists: {list(models.keys())}")
        return models[model_name]

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, model_name: str = "linear_regression"):
        self.model_name = model_name
        self.model = self._get_model(model_name)

        self.log.info(f"{model_name} train started")
        self.model.fit(x_train, y_train)
        self.log.info(f"{model_name} train finished")

        return self

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        if self.model is None:
            raise ValueError("Firstly, Execute train()")

        y_pred = self.model.predict(x_test)

        metrics = {
            "model": self.model_name,
            "version": self.version,
            "r2": round(r2_score(y_test, y_pred), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        }

        self.log.info(f"Evaluate: {metrics}")
        return metrics

    def save_model(self):
        if self.model is None:
            raise ValueError("Firstly, Execute train()")

        os.makedirs(self.models_dir, exist_ok=True)
        path = os.path.join(self.models_dir, f"{self.model_name}.joblib")
        joblib.dump(self.model, path)
        self.log.info(f"Model saved: {path}")
    
    def save_best_model(self):
        path = os.path.join(self.results_dir, "metrics.csv")

        if not os.path.exists(path):
            raise FileNotFoundError("Firstly, Execute save_results()")

        df = pd.read_csv(path)
        best = df.loc[df["r2"].idxmax()]

        best_model_name = best["model"]
        self.load_model(best_model_name)

        os.makedirs(self.models_dir, exist_ok=True)
        save_path = os.path.join(self.models_dir, "best_model.joblib")
        joblib.dump(self.model, save_path)

        self.log.info(f"Best model: {best_model_name} | R2: {best['r2']} | saved: {save_path}")
        return best.to_dict()

    def save_results(self, metrics: dict):
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, "metrics.csv")

        df_new = pd.DataFrame([metrics])

        if os.path.exists(path):
            df_old = pd.read_csv(path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(path, index=False)
        
        self.log.info(f"Results saved: {path}")

    def load_model(self, model_name: str):
        path = os.path.join(self.models_dir, f"{model_name}.joblib")
        self.model = joblib.load(path)
        self.model_name = model_name
        self.log.info(f"Model loaded: {path}")
        return self
    
    def shap_explain(self, x_train: pd.DataFrame, x_test: pd.DataFrame):

        if self.model is None:
            raise ValueError("Firstly, execute train()")

        self.log.info(f"SHAP explain started: {self.model_name}")

        if self.model_name in ["random_forest"]:
            explainer = shap.TreeExplainer(self.model, x_train, feature_names=x_train.columns.tolist())
        else:
            explainer = shap.LinearExplainer(self.model, x_train, feature_names=x_train.columns.tolist())

        x_sample = x_test.iloc[:500]
        shap_values = explainer(x_sample)

        print(f"\n📊 SHAP Summary — {self.model_name}")
        shap.summary_plot(shap_values, x_sample)
        shap.summary_plot(shap_values, x_sample, plot_type="bar")
        
        # shap.plots.bar(shap_values, max_display=10)
        # plt.show()

        self.log.info("SHAP explain finished")
        return shap_values
