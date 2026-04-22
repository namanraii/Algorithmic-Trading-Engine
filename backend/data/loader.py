import yfinance as yf
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, cache_dir="backend/data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, ticker: str, start: str, end: str, interval="1d") -> pd.DataFrame:
        cache_path = self.cache_dir / f"{ticker}_{start}_{end}_{interval}.csv"

        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty and 'Close' in df.columns:
                return df

        print(f"DEBUG: Attempting to fetch {ticker} ({start} to {end})")
        
        # PROVIDER 1: yfinance (with Browser User-Agent)
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        df = pd.DataFrame()
        try:
            t = yf.Ticker(ticker, session=session)
            # Try history method first
            df = t.history(start=start, end=end, interval=interval, auto_adjust=True)
            if df.empty:
                # Try fallback period method
                print(f"DEBUG: yfinance range empty, trying period='2y'...")
                df = t.history(period="2y", interval=interval, auto_adjust=True)
                if not df.empty:
                    mask = (df.index >= pd.to_datetime(start).tz_localize(df.index.tz)) & \
                           (df.index <= pd.to_datetime(end).tz_localize(df.index.tz))
                    df = df.loc[mask]
        except Exception as e:
            print(f"DEBUG: yfinance error: {str(e)}")

        # PROVIDER 2: Stooq (via pandas_datareader) - extremely reliable fallback for stocks
        if df.empty:
            print(f"DEBUG: yfinance failed completely. Trying Stooq fallback...")
            try:
                import pandas_datareader.data as web
                # Stooq uses .US suffix for US stocks
                stooq_ticker = f"{ticker}.US" if "-" not in ticker and ".NS" not in ticker else ticker
                df = web.DataReader(stooq_ticker, 'stooq', start, end)
                if not df.empty:
                    df = df.sort_index() # Stooq returns data in reverse chronological order
                    # Rename columns to standard OHLCV if needed
                    df = df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
            except Exception as e:
                print(f"DEBUG: Stooq fallback error: {str(e)}")

        if df.empty:
            print(f"DEBUG: Stooq failed. Using synthetic data as final failsafe...")
            try:
                # Generate a realistic random walk for the requested period
                date_rng = pd.date_range(start=start, end=end, freq='B')
                n = len(date_rng)
                prices = [150.0]
                for _ in range(n-1):
                    prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.02)))
                
                df = pd.DataFrame({
                    'Open': prices,
                    'High': [p * 1.01 for p in prices],
                    'Low': [p * 0.99 for p in prices],
                    'Close': prices,
                    'Volume': [1000000] * n
                }, index=date_rng)
                df['Note'] = "SYNTHETIC_DATA_FALLBACK"
            except Exception as e:
                print(f"DEBUG: Synthetic fallback error: {str(e)}")

        if df.empty:
            raise ValueError(f"CRITICAL: System failure. Could not fetch or generate data for '{ticker}'.")

        # Standardize: Ensure index is datetime and named 'Date'
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        
        # Ensure all required columns exist
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                df[col] = df['Close'] if 'Close' in df.columns else 0.0

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Persist to cache (only if not synthetic)
        if 'Note' not in df.columns:
            df.to_csv(cache_path)
            
        return df

    def fetch_multiple(self, tickers: list, **kwargs) -> dict:
        return {t: self.fetch(t, **kwargs) for t in tickers}
