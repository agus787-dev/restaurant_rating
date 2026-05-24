import pandas as pd
from src.logger import Logger


class FeatureEngineer:
    def __init__(self):
        self.log = Logger("feature_engineer")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.info("Feature engineering started")
        df = df.copy()

        # Price
        df["cost_per_person"] = df["average_cost_for_two"] / 2
        self.log.info("cost_per_person added")

        # Active
        df["is_fully_active"] = (
            df["has_table_booking"] +
            df["has_online_delivery"] +
            df["is_delivering_now"]
        )
        self.log.info("is_fully_active added")

        # Location
        df["lat_lon_interaction"] = df["latitude"] * df["longitude"]
        self.log.info("lat_lon_interaction added")

        # # vote / price
        # df["votes_per_price"] = df["votes"] / (df["price_range"] + 1)
        # self.log.info("votes_per_price added")

        self.log.info(f"Feature engineering finished: {df.shape[1]} columns")
        return df