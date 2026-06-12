import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import shap
import warnings
import sys
import subprocess

warnings.filterwarnings('ignore')

def _check_tensorflow():
    try:
        cmd = [sys.executable, "-c", "import tensorflow"]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return res.returncode == 0
    except Exception:
        return False

HAS_TENSORFLOW = _check_tensorflow()

if HAS_TENSORFLOW:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
else:
    print("Warning: TensorFlow is unavailable or crashed during import. LSTM forecaster will use a LinearRegression fallback.")

class LSTMForecaster:
    def __init__(self, sequence_length=60):
        self.seq_len = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self._is_fallback = not HAS_TENSORFLOW
    
    def _build_model(self, n_features):
        if self._is_fallback:
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        
        model = Sequential([
            Input(shape=(self.seq_len, n_features)),
            LSTM(64, return_sequences=True), # Reduced from 128 for performance
            Dropout(0.2),
            LSTM(32, return_sequences=False), # Reduced from 64 for performance
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1)  # predict next close price
        ])
        model.compile(optimizer='adam', loss='huber')
        return model
    
    def prepare_sequences(self, df: pd.DataFrame, feature_cols: list):
        if self._is_fallback:
            return None, None
        data = self.scaler.fit_transform(df[feature_cols + ['Close']].values)
        X, y = [], []
        for i in range(self.seq_len, len(data)):
            X.append(data[i-self.seq_len:i, :-1])  # features
            y.append(data[i, -1])                   # close price
        return np.array(X), np.array(y)
    
    def train(self, df, feature_cols, epochs=1, validation_split=0.1): 
        if self._is_fallback:
            from sklearn.linear_model import LinearRegression
            X = df[feature_cols].dropna()
            y = df['Close'].shift(-1).loc[X.index].fillna(df['Close'])
            if not X.empty:
                self.model = LinearRegression()
                self.model.fit(X, y)
            return None
            
        X, y = self.prepare_sequences(df, feature_cols)
        if len(X) == 0:
            return None
        split = int(len(X) * 0.8)
        self.model = self._build_model(len(feature_cols))
        history = self.model.fit(
            X[:split], y[:split],
            validation_data=(X[split:], y[split:]),
            epochs=epochs, batch_size=64, verbose=0 # Increased batch size, reduced epochs
        )
        return history
        
    def predict(self, df, feature_cols):
        if self._is_fallback:
            if self.model is None:
                self.train(df, feature_cols)
            X = df[feature_cols].dropna()
            if X.empty or self.model is None:
                return df['Close'].values
            predictions = self.model.predict(X)
            return predictions

        if self.model is None:
            self.train(df, feature_cols)
        X, _ = self.prepare_sequences(df, feature_cols)
        if len(X) == 0:
            return np.array([])
        predictions = self.model.predict(X, verbose=0)
        
        # Need to inverse transform. We need a dummy array of the same shape as the scaler's input
        dummy = np.zeros((len(predictions), len(feature_cols) + 1))
        dummy[:, -1] = predictions[:, 0]
        return self.scaler.inverse_transform(dummy)[:, -1]

class XGBoostSignalClassifier:
    def __init__(self, threshold=0.01):
        self.threshold = threshold  # 1% price move = positive class
        self.model = None
    
    def prepare_labels(self, df: pd.DataFrame) -> pd.Series:
        future_return = df['Close'].shift(-1) / df['Close'] - 1
        return (future_return > self.threshold).astype(int)
    
    def train(self, df: pd.DataFrame, feature_cols: list):
        X = df[feature_cols].dropna()
        y = self.prepare_labels(df).loc[X.index]
        
        # TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3) # Reduced splits for performance
        
        self.model = xgb.XGBClassifier(
            n_estimators=20, learning_rate=0.2, # Further reduced n_estimators
            max_depth=3, subsample=0.8,
            colsample_bytree=0.8, eval_metric='logloss'
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.model.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                verbose=False
            )
            
        return self.model
    
    def predict(self, df, feature_cols):
        X = df[feature_cols].dropna()
        if self.model is None:
            self.train(df, feature_cols)
        return self.model.predict(X)
        
    def explain(self, X):
        if self.model is None:
            return None
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X)
