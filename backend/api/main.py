from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import datetime
import sys
import os

# Prevent macOS Segfault with XGBoost/Tensorflow OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.data.loader import DataLoader
from backend.utils.feature_engineer import FeatureEngineer
from backend.strategies.base import EMACrossoverStrategy, RSIMeanReversionStrategy, MACDStrategy
from backend.backtester.engine import BacktestEngine
from backend.backtester.metrics import MetricsCalculator

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="Algo Trading Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class BacktestRequest(BaseModel):
    ticker: str
    strategy: str  # 'ema', 'rsi', 'macd'
    start_date: str
    end_date: str
    initial_capital: float = 100000


def serialize(obj):
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


@app.get("/")
def root():
    return {"status": "Algo Trading Engine API is running 🚀"}


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    try:
        loader = DataLoader()
        df = loader.fetch(req.ticker, req.start_date, req.end_date)

        if df.empty:
            raise HTTPException(status_code=400, detail="No data found for the given ticker and date range.")

        fe = FeatureEngineer()
        df = fe.add_all_indicators(df)

        if df.empty:
            raise HTTPException(status_code=400, detail="Not enough data to compute indicators.")

        strategy_map = {
            'ema': EMACrossoverStrategy(),
            'rsi': RSIMeanReversionStrategy(),
            'macd': MACDStrategy()
        }

        if req.strategy not in strategy_map:
            raise HTTPException(status_code=400, detail="Invalid strategy. Choose: ema, rsi, macd")

        strategy = strategy_map[req.strategy]
        signals = strategy.generate_signals(df)
        engine = BacktestEngine(initial_capital=req.initial_capital)
        result = engine.run(df, signals)

        metrics = MetricsCalculator().calculate_all(
            result['portfolio']['value'],
            result['trades']
        )

        portfolio_records = result['portfolio'].reset_index().to_dict('records')
        trades_records = result['trades'].to_dict('records') if not result['trades'].empty else []

        return serialize({
            'portfolio': portfolio_records,
            'trades': trades_records,
            'metrics': metrics
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compare")
def compare_strategies(ticker: str, start: str, end: str):
    try:
        loader = DataLoader()
        df = loader.fetch(ticker, start, end)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found.")

        fe = FeatureEngineer()
        df = fe.add_all_indicators(df)

        strategies = {
            'ema': EMACrossoverStrategy(),
            'rsi': RSIMeanReversionStrategy(),
            'macd': MACDStrategy()
        }

        comparison = {}
        for name, strategy in strategies.items():
            signals = strategy.generate_signals(df)
            engine = BacktestEngine()
            result = engine.run(df, signals)
            metrics = MetricsCalculator().calculate_all(
                result['portfolio']['value'], result['trades']
            )
            comparison[name] = metrics

        return serialize(comparison)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/{ticker}")
def get_ml_prediction(ticker: str):
    print(f"DEBUG: ML prediction requested for {ticker} (Mocked)")
    return {
        "ticker": ticker,
        "current_price": 150.0,
        "predicted_price_lstm": 155.0,
        "signal_xgboost": "BUY",
        "date": str(datetime.date.today()),
        "note": "ML is currently mocked for stability"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
