from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def _clean_signals(self, df):
        df['position'] = df['signal'].diff()
        df['position'] = df['position'].fillna(0)
        return df

class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, fast=20, slow=50):
        super().__init__("EMA Crossover")
        self.fast = fast
        self.slow = slow
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'ema_fast' not in df.columns or 'ema_slow' not in df.columns:
            df['ema_fast'] = df['Close'].ewm(span=self.fast).mean()
            df['ema_slow'] = df['Close'].ewm(span=self.slow).mean()
        
        df['signal'] = 0
        df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] = 1
        df.loc[df['ema_fast'] < df['ema_slow'], 'signal'] = -1
        
        return self._clean_signals(df)

class RSIMeanReversionStrategy(BaseStrategy):
    def __init__(self, oversold=30, overbought=70, window=14):
        super().__init__("RSI Mean Reversion")
        self.oversold = oversold
        self.overbought = overbought
        self.window = window
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'rsi' not in df.columns:
            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(self.window).mean()
            loss = -delta.clip(upper=0).rolling(self.window).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        df['signal'] = 0
        df.loc[df['rsi'] < self.oversold, 'signal'] = 1
        df.loc[df['rsi'] > self.overbought, 'signal'] = -1
        
        return self._clean_signals(df)

class MACDStrategy(BaseStrategy):
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__("MACD Momentum")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            ema_fast = df['Close'].ewm(span=self.fast).mean()
            ema_slow = df['Close'].ewm(span=self.slow).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=self.signal_period).mean()
        
        df['signal'] = 0
        df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1
        df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1
        
        return self._clean_signals(df)
