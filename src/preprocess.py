import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.logger import Logger

class Preprocessor:
    def __init__(self):
        self.scaler   = StandardScaler()
        self.encoders = {}
        self.onehot_cols = []
        self.fill_values = {}
        self.scalers = {}
        
        self.log = Logger("preprocessor")
    
    def fillna_train(self, df):
        self.log.info("fillna_train started")   
        
        df=df.copy()

        for col in df.columns:
            if df[col].dtype == 'object':
                self.fill_values[col] = df[col].mode()[0]
                df[col].fillna(self.fill_values[col], inplace=True)
            else:
                self.fill_values[col] = df[col].mean()
                df[col].fillna(self.fill_values[col], inplace=True)
                
        self.log.info(f"fillna_train finished: {len(self.fill_values)} columns filled")
        
        return df

    def fillna_test(self, df):
        self.log.info("fillna_test started")
        df=df.copy()

        for col, value in self.fill_values.items():
            df[col].fillna(value, inplace=True)
        self.log.info("fillna_test finished")
        return df
    
    def apply_fillna_train_to_test(self, x_train, x_test):
        x_train_filled = self.fillna_train(df=x_train)
        x_test_filled = self.fillna_test(df=x_test)
        
        return x_train_filled, x_test_filled
    
    def encode_train(self, df, threshold=0):
        self.log.info("encode_train started")
        df = df.copy()

        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].nunique() <= threshold:
                    self.onehot_cols.append(col)
                    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                    df = pd.concat([df.drop(columns=col), dummies], axis=1)
                    self.log.info(f"  One-hot encoding: {col}")
                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.encoders[col] = le
                    self.log.info(f"  Label encoding: {col}")
        self.log.info(f"encode_train finished: {len(self.encoders)} label, {len(self.onehot_cols)} onehot")
        return df

    def encode_test(self, df, train_columns):
        self.log.info("encode_test started")
        df = df.copy()

        # Label encoding
        for col, le in self.encoders.items():
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            self.log.info(f"  Label encoding: {col}")

        # One-hot ONLY for known columns
        for col in self.onehot_cols:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df.drop(columns=col), dummies], axis=1)
            self.log.info(f"  One-hot encoding: {col}")

        # Align columns
        df = df.reindex(columns=train_columns, fill_value=0)
        self.log.info(f"encode_test finished: {df.shape[1]} columns")

        return df
    
    def apply_encode_train_to_test(self, x_train, x_test):
        
        x_train_encoded = self.encode_train(df=x_train)
        x_test_encoded = self.encode_test(
            df=x_test, 
            train_columns=x_train_encoded.columns
        )
        
        return x_train_encoded, x_test_encoded
    
    def scale_train(self, df):
        self.log.info("scale_train started")
        df = df.copy()

        for col in df.columns:
            if df[col].dtype != 'object':
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
        self.log.info(f"scale_train finished: {len(self.scalers)} columns are done scaled")
        return df

    def scale_test(self, df):
        self.log.info("scale_test started")
        df = df.copy()

        for col in df.columns:
            if col in self.scalers:
                df[col] = self.scalers[col].transform(df[[col]])
        self.log.info("scale_test finished")
        return df
    
    def apply_scale_train_to_test(self, x_train, x_test):
        x_train_scaled = self.scale_train(df=x_train)
        x_test_scaled = self.scale_test(df=x_test)
        
        return x_train_scaled, x_test_scaled