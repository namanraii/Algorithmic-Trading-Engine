"""
Backfill script for the Algorithmic Trading Engine.

Replays backtests across 1-2 years for 3-5 stocks to populate:
  - price_history
  - backtest_runs
  - trades
  - portfolio_snapshots
  - model_predictions

Run with:
    cd backend && python scripts/backfill.py

Requires:
    - Postgres running (local or via docker-compose)
    - DATABASE_URL env var set (or defaults to localhost)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import datetime
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal, engine, Base
from backend.db import crud
from backend.data.loader import DataLoader
from backend.utils.feature_engineer import FeatureEngineer
from backend.strategies.base import EMACrossoverStrategy, RSIMeanReversionStrategy, MACDStrategy
from backend.backtester.engine import BacktestEngine
from backend.backtester.metrics import MetricsCalculator
from backend.ml_models.models import LSTMForecaster, XGBoostSignalClassifier

# ── Configuration ───────────────────────────────────────────────────

TICKERS = [
    ("AAPL", "Apple Inc.", "NASDAQ"),
    ("TSLA", "Tesla Inc.", "NASDAQ"),
    ("NVDA", "NVIDIA Corp.", "NASDAQ"),
    ("MSFT", "Microsoft Corp.", "NASDAQ"),
    ("GOOGL", "Alphabet Inc.", "NASDAQ"),
]

STRATEGIES = {
    'ema': EMACrossoverStrategy(),
    'rsi': RSIMeanReversionStrategy(),
    'macd': MACDStrategy(),
}

START_DATE = "2022-01-01"
END_DATE = "2023-12-31"
INITIAL_CAPITAL = 100000.0


def backfill_backtests(db: Session):
    """Run backtests for all ticker + strategy combinations and persist."""
    loader = DataLoader()
    fe = FeatureEngineer()
    metrics_calc = MetricsCalculator()

    for ticker, name, exchange in TICKERS:
        print(f"\n📊 Fetching data for {ticker}...")
        try:
            df = loader.fetch(ticker, START_DATE, END_DATE)
            if df.empty or len(df) < 60:
                print(f"  ⚠️ Insufficient data for {ticker}, skipping.")
                continue
        except Exception as e:
            print(f"  ❌ Failed to fetch {ticker}: {e}")
            continue

        # Ensure stock record exists
        stock = crud.get_or_create_stock(db, ticker, name=name, exchange=exchange)

        # Save price history
        print(f"  💾 Saving price history ({len(df)} rows)...")
        crud.save_price_history(db, stock.id, df[['Open', 'High', 'Low', 'Close', 'Volume']])

        # Compute indicators once
        try:
            df_ind = fe.add_all_indicators(df)
        except Exception as e:
            print(f"  ⚠️ Indicator computation failed for {ticker}: {e}")
            continue

        for strat_name, strategy in STRATEGIES.items():
            print(f"  🔬 Running {strat_name.upper()} strategy...")
            try:
                signals = strategy.generate_signals(df_ind)
                engine_bt = BacktestEngine(initial_capital=INITIAL_CAPITAL)
                result = engine_bt.run(df_ind, signals)

                metrics = metrics_calc.calculate_all(
                    result['portfolio']['value'],
                    result['trades']
                )

                start_d = datetime.datetime.strptime(START_DATE, "%Y-%m-%d").date()
                end_d = datetime.datetime.strptime(END_DATE, "%Y-%m-%d").date()
                final_cap = float(result['final_capital']) if not hasattr(result['final_capital'], 'item') else result['final_capital'].item()

                # Persist backtest run
                backtest_run = crud.create_backtest_run(
                    db, stock.id, strat_name, start_d, end_d,
                    INITIAL_CAPITAL, final_cap, metrics
                )

                # Persist trades
                crud.save_trades(db, backtest_run.id, stock.id, result['trades'])

                # Persist portfolio snapshots
                crud.save_portfolio_snapshots(
                    db, backtest_run.id,
                    result['portfolio'][['value']],
                    result['portfolio'][['benchmark']] if 'benchmark' in result['portfolio'].columns else None
                )

                print(f"    ✅ Saved backtest #{backtest_run.id} — Return: {metrics['total_return']:.2f}%, Trades: {metrics['total_trades']}")

            except Exception as e:
                print(f"    ❌ Backtest failed for {ticker}/{strat_name}: {e}")
                continue


def backfill_predictions(db: Session):
    """Generate ML predictions for each ticker and persist."""
    feature_cols = ['ema_20', 'ema_50', 'macd', 'macd_signal', 'rsi', 'bb_upper', 'bb_lower', 'atr', 'obv']

    for ticker, name, exchange in TICKERS:
        print(f"\n🤖 Generating ML predictions for {ticker}...")
        stock = crud.get_or_create_stock(db, ticker, name=name, exchange=exchange)

        # Fetch 2 years of data
        end = datetime.date.today()
        start = end - datetime.timedelta(days=730)

        loader = DataLoader()
        try:
            df = loader.fetch(ticker, start.isoformat(), end.isoformat())
            if df.empty or len(df) < 60:
                print(f"  ⚠️ Insufficient data for ML on {ticker}")
                continue
        except Exception as e:
            print(f"  ❌ Data fetch failed for {ticker}: {e}")
            continue

        fe = FeatureEngineer()
        try:
            df = fe.add_all_indicators(df)
        except Exception as e:
            print(f"  ⚠️ Feature engineering failed for {ticker}: {e}")
            continue

        available_features = [c for c in feature_cols if c in df.columns]
        if len(available_features) < 3:
            print(f"  ⚠️ Not enough features for {ticker}")
            continue

        # LSTM prediction
        try:
            lstm = LSTMForecaster(sequence_length=60)
            lstm_preds = lstm.predict(df, available_features)
            if len(lstm_preds) > 0:
                predicted_price = float(lstm_preds[-1])
                actual_price = float(df['Close'].iloc[-1])
                pred = crud.create_model_prediction(
                    db, stock.id, "LSTM",
                    predicted_value=predicted_price,
                    prediction_date=end
                )
                # Immediately fill actual price for accuracy
                crud.update_prediction_actual(db, pred.id, actual_price)
                print(f"    ✅ LSTM prediction saved: ${predicted_price:.2f} (actual: ${actual_price:.2f})")
        except Exception as e:
            print(f"    ❌ LSTM failed for {ticker}: {e}")

        # XGBoost prediction
        try:
            xgb = XGBoostSignalClassifier(threshold=0.01)
            xgb_signals = xgb.predict(df, available_features)
            if len(xgb_signals) > 0:
                signal = "BUY" if xgb_signals[-1] == 1 else "HOLD / SELL"
                crud.create_model_prediction(
                    db, stock.id, "XGBoost",
                    predicted_signal=signal,
                    prediction_date=end
                )
                print(f"    ✅ XGBoost signal saved: {signal}")
        except Exception as e:
            print(f"    ❌ XGBoost failed for {ticker}: {e}")


def main():
    print("=" * 60)
    print("  ALGORITHMIC TRADING ENGINE — DATABASE BACKFILL")
    print("=" * 60)

    # Ensure tables exist
    print("\n🔧 Ensuring database tables exist...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        print("\n📈 Phase 1: Backfilling backtest runs, trades, and portfolio snapshots...")
        backfill_backtests(db)

        print("\n🤖 Phase 2: Backfilling model predictions...")
        backfill_predictions(db)

        print("\n" + "=" * 60)
        print("  BACKFILL COMPLETE ✅")
        print("=" * 60)

        # Summary
        from backend.db.models import BacktestRun, Trade, ModelPrediction
        bt_count = db.query(BacktestRun).count()
        trade_count = db.query(Trade).count()
        pred_count = db.query(ModelPrediction).count()

        print(f"\n📊 Database Summary:")
        print(f"   Backtest runs:   {bt_count}")
        print(f"   Trades:          {trade_count}")
        print(f"   Predictions:     {pred_count}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
