"""CRUD helpers for the trading engine database."""

from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc
from typing import Optional, List, Dict, Any
from datetime import date, datetime
import pandas as pd

from backend.db.models import (
    Stock, PriceHistory, BacktestRun, Trade,
    PortfolioSnapshot, ModelPrediction
)


# ── Stocks ─────────────────────────────────────────────────────────

def get_or_create_stock(db: Session, ticker: str, name: str = None, exchange: str = None) -> Stock:
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    if not stock:
        stock = Stock(ticker=ticker, name=name, exchange=exchange)
        db.add(stock)
        db.commit()
        db.refresh(stock)
    return stock


def get_stock_by_ticker(db: Session, ticker: str) -> Optional[Stock]:
    return db.query(Stock).filter(Stock.ticker == ticker).first()


# ── Price History ──────────────────────────────────────────────────

def get_price_history(db: Session, stock_id: int, start: date, end: date) -> pd.DataFrame:
    rows = db.query(PriceHistory).filter(
        PriceHistory.stock_id == stock_id,
        PriceHistory.date >= start,
        PriceHistory.date <= end
    ).order_by(asc(PriceHistory.date)).all()
    
    if not rows:
        return pd.DataFrame()
    
    data = [{
        'Date': r.date,
        'Open': r.open,
        'High': r.high,
        'Low': r.low,
        'Close': r.close,
        'Volume': r.volume,
    } for r in rows]
    
    df = pd.DataFrame(data)
    if not df.empty:
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
    return df


def save_price_history(db: Session, stock_id: int, df: pd.DataFrame):
    """Upsert price history from a DataFrame."""
    if df.empty:
        return
    
    # Get existing dates to avoid duplicates
    existing = db.query(PriceHistory.date).filter(
        PriceHistory.stock_id == stock_id
    ).all()
    existing_dates = {r[0] for r in existing}
    
    records = []
    for idx, row in df.iterrows():
        d = pd.to_datetime(idx).date() if hasattr(idx, 'date') else idx
        if d in existing_dates:
            continue
        records.append(PriceHistory(
            stock_id=stock_id,
            date=d,
            open=float(row.get('Open', row.get('open', 0))),
            high=float(row.get('High', row.get('high', 0))),
            low=float(row.get('Low', row.get('low', 0))),
            close=float(row.get('Close', row.get('close', 0))),
            volume=float(row.get('Volume', row.get('volume', 0))),
        ))
    
    if records:
        db.bulk_save_objects(records)
        db.commit()


# ── Backtest Runs ─────────────────────────────────────────────────

def get_cached_backtest(
    db: Session, stock_id: int, strategy: str,
    start: date, end: date, initial_capital: float
) -> Optional[BacktestRun]:
    """Return a recent backtest run if one exists for the exact parameters."""
    return db.query(BacktestRun).filter(
        BacktestRun.stock_id == stock_id,
        BacktestRun.strategy == strategy,
        BacktestRun.start_date == start,
        BacktestRun.end_date == end,
        BacktestRun.initial_capital == initial_capital,
    ).order_by(desc(BacktestRun.created_at)).first()


def create_backtest_run(
    db: Session, stock_id: int, strategy: str,
    start: date, end: date, initial_capital: float,
    final_capital: float, metrics: Dict[str, Any]
) -> BacktestRun:
    run = BacktestRun(
        stock_id=stock_id,
        strategy=strategy,
        start_date=start,
        end_date=end,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=metrics.get('total_return'),
        sharpe_ratio=metrics.get('sharpe_ratio'),
        max_drawdown=metrics.get('max_drawdown'),
        calmar_ratio=metrics.get('calmar_ratio'),
        win_rate=metrics.get('win_rate'),
        total_trades=metrics.get('total_trades'),
        volatility=metrics.get('volatility'),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_backtest_runs(db: Session, stock_id: int = None, strategy: str = None, limit: int = 50) -> List[BacktestRun]:
    q = db.query(BacktestRun)
    if stock_id:
        q = q.filter(BacktestRun.stock_id == stock_id)
    if strategy:
        q = q.filter(BacktestRun.strategy == strategy)
    return q.order_by(desc(BacktestRun.created_at)).limit(limit).all()


# ── Trades ────────────────────────────────────────────────────────

def save_trades(db: Session, backtest_run_id: int, stock_id: int, trades_df: pd.DataFrame):
    if trades_df.empty:
        return
    
    records = []
    buy_queue = []  # FIFO queue of (price, shares) for PnL matching
    
    for _, row in trades_df.iterrows():
        t_type = row.get('type', 'BUY')
        price = float(row.get('price', 0))
        shares = int(row.get('shares', 0))
        order_value = price * shares
        pnl = None
        
        if t_type == 'SELL':
            # Match with earliest BUYs (FIFO) for PnL
            if buy_queue:
                matched_shares = 0
                total_cost = 0
                remaining = shares
                
                while remaining > 0 and buy_queue:
                    buy_price, buy_shares = buy_queue[0]
                    take = min(remaining, buy_shares)
                    total_cost += take * buy_price
                    matched_shares += take
                    remaining -= take
                    buy_queue[0] = (buy_price, buy_shares - take)
                    if buy_queue[0][1] == 0:
                        buy_queue.pop(0)
                
                if matched_shares > 0:
                    avg_cost = total_cost / matched_shares
                    pnl = (price - avg_cost) * matched_shares
        else:
            buy_queue.append((price, shares))
        
        date_val = row.get('date')
        if isinstance(date_val, pd.Timestamp):
            date_val = date_val.date()
        elif isinstance(date_val, str):
            date_val = pd.to_datetime(date_val).date()
        elif hasattr(date_val, 'date'):
            date_val = date_val.date()
        
        records.append(Trade(
            backtest_run_id=backtest_run_id,
            stock_id=stock_id,
            date=date_val,
            type=t_type,
            price=price,
            shares=shares,
            order_value=order_value,
            pnl=pnl,
        ))
    
    if records:
        db.bulk_save_objects(records)
        db.commit()


def get_trades_by_backtest(db: Session, backtest_run_id: int) -> List[Trade]:
    return db.query(Trade).filter(
        Trade.backtest_run_id == backtest_run_id
    ).order_by(asc(Trade.date)).all()


def get_top_profitable_trades(db: Session, limit: int = 5) -> List[Trade]:
    return db.query(Trade).filter(
        Trade.type == 'SELL',
        Trade.pnl != None
    ).order_by(desc(Trade.pnl)).limit(limit).all()


# ── Portfolio Snapshots ─────────────────────────────────────────────

def save_portfolio_snapshots(
    db: Session, backtest_run_id: int,
    portfolio_df: pd.DataFrame, benchmark_df: pd.DataFrame = None
):
    if portfolio_df.empty:
        return
    
    records = []
    for idx, row in portfolio_df.iterrows():
        date_val = idx
        if isinstance(date_val, pd.Timestamp):
            date_val = date_val.date()
        elif isinstance(date_val, str):
            date_val = pd.to_datetime(date_val).date()
        elif hasattr(date_val, 'date'):
            date_val = date_val.date()
        
        strategy_value = float(row.get('value', row.get('strategy_value', 0)))
        benchmark_value = strategy_value  # default
        
        if benchmark_df is not None and not benchmark_df.empty:
            if idx in benchmark_df.index:
                benchmark_value = float(benchmark_df.loc[idx].get('benchmark', benchmark_value))
        
        records.append(PortfolioSnapshot(
            backtest_run_id=backtest_run_id,
            date=date_val,
            strategy_value=strategy_value,
            benchmark_value=benchmark_value,
        ))
    
    if records:
        db.bulk_save_objects(records)
        db.commit()


def get_portfolio_snapshots(db: Session, backtest_run_id: int) -> pd.DataFrame:
    rows = db.query(PortfolioSnapshot).filter(
        PortfolioSnapshot.backtest_run_id == backtest_run_id
    ).order_by(asc(PortfolioSnapshot.date)).all()
    
    if not rows:
        return pd.DataFrame()
    
    data = [{
        'date': r.date,
        'value': r.strategy_value,
        'benchmark': r.benchmark_value,
    } for r in rows]
    
    df = pd.DataFrame(data)
    if not df.empty:
        df.set_index('date', inplace=True)
    return df


# ── Model Predictions ─────────────────────────────────────────────

def create_model_prediction(
    db: Session, stock_id: int, model_type: str,
    predicted_value: float = None, predicted_signal: str = None,
    prediction_date: date = None, horizon_days: int = 1
) -> ModelPrediction:
    pred = ModelPrediction(
        stock_id=stock_id,
        model_type=model_type,
        predicted_value=predicted_value,
        predicted_signal=predicted_signal,
        prediction_date=prediction_date or date.today(),
        horizon_days=horizon_days,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def update_prediction_actual(
    db: Session, prediction_id: int, actual_price: float
):
    pred = db.query(ModelPrediction).filter(ModelPrediction.id == prediction_id).first()
    if pred and pred.predicted_value:
        pred.actual_price = actual_price
        pred.accuracy = abs(pred.predicted_value - actual_price) / actual_price
        db.commit()
        db.refresh(pred)
    return pred


def get_predictions_by_stock(
    db: Session, stock_id: int, model_type: str = None, limit: int = 100
) -> List[ModelPrediction]:
    q = db.query(ModelPrediction).filter(ModelPrediction.stock_id == stock_id)
    if model_type:
        q = q.filter(ModelPrediction.model_type == model_type)
    return q.order_by(desc(ModelPrediction.prediction_date)).limit(limit).all()


def get_prediction_accuracy_over_time(
    db: Session, stock_id: int, model_type: str
) -> List[Dict[str, Any]]:
    rows = db.query(ModelPrediction).filter(
        ModelPrediction.stock_id == stock_id,
        ModelPrediction.model_type == model_type,
        ModelPrediction.accuracy != None
    ).order_by(asc(ModelPrediction.prediction_date)).all()
    
    return [{
        'date': r.prediction_date.isoformat(),
        'predicted': r.predicted_value,
        'actual': r.actual_price,
        'accuracy': r.accuracy,
    } for r in rows]
