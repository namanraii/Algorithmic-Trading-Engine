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
            # Validate cache
            if not df.empty and 'Close' in df.columns:
                return df

        print(f"DEBUG: Fetching data for {ticker} from {start} to {end}")
        
        # Use a custom session/headers to avoid being blocked by Yahoo Finance on cloud IPs
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        })

        # Download fresh — auto_adjust=True gives clean OHLCV without MultiIndex
        df = yf.download(ticker, start=start, end=end, interval=interval,
                         auto_adjust=True, progress=False, session=session)

        if df.empty:
            print(f"ERROR: yfinance returned empty DataFrame for {ticker}")
            raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

        # Flatten MultiIndex if present (yfinance 0.2.40+ often returns this)
        if isinstance(df.columns, pd.MultiIndex):
            # Try to find the OHLCV levels
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(1)

        # Persist to cache
        df.to_csv(cache_path)
        return df

    def fetch_multiple(self, tickers: list, **kwargs) -> dict:
        return {t: self.fetch(t, **kwargs) for t in tickers}
