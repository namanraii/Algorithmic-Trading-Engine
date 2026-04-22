import pandas as pd
import ta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Flatten MultiIndex columns — yfinance v0.2+ returns MultiIndex for some tickers
        if isinstance(df.columns, pd.MultiIndex):
            # Take only the first level (Open, High, Low, Close, Volume)
            df.columns = df.columns.get_level_values(0)

        # Deduplicate columns (can happen with MultiIndex flattening)
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # Strip any whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}. Got: {list(df.columns)}")

        # Make sure all price columns are numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where Close is NaN
        df = df.dropna(subset=['Close'])

        if len(df) < 60:
            raise ValueError(f"Not enough data: only {len(df)} rows after cleaning. Need at least 60.")

        # Trend indicators
        df['ema_20']     = ta.trend.ema_indicator(df['Close'], window=20)
        df['ema_50']     = ta.trend.ema_indicator(df['Close'], window=50)
        df['macd']       = ta.trend.macd(df['Close'])
        df['macd_signal']= ta.trend.macd_signal(df['Close'])
        df['adx']        = ta.trend.adx(df['High'], df['Low'], df['Close'])

        # Momentum
        df['rsi']        = ta.momentum.rsi(df['Close'], window=14)
        df['stoch_k']    = ta.momentum.stoch(df['High'], df['Low'], df['Close'])

        # Volatility
        df['bb_upper']   = ta.volatility.bollinger_hband(df['Close'])
        df['bb_lower']   = ta.volatility.bollinger_lband(df['Close'])
        df['atr']        = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Volume
        df['obv']        = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['vwap']       = ta.volume.volume_weighted_average_price(
                               df['High'], df['Low'], df['Close'], df['Volume'])

        df = df.dropna()

        if len(df) < 10:
            raise ValueError("Too many NaN values after computing indicators. Try a longer date range.")

        return df
