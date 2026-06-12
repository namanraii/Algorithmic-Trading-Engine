"""SQLAlchemy ORM models for the Algorithmic Trading Engine."""

from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime,
    ForeignKey, Index, Text, Enum, Boolean
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.db.database import Base
import enum


class StrategyEnum(str, enum.Enum):
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"


class TradeType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class ModelType(str, enum.Enum):
    LSTM = "LSTM"
    XGBOOST = "XGBoost"


# ── stocks ──────────────────────────────────────────────────────────
class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=True)
    exchange = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    price_history = relationship("PriceHistory", back_populates="stock", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="stock", cascade="all, delete-orphan")
    backtest_runs = relationship("BacktestRun", back_populates="stock", cascade="all, delete-orphan")
    model_predictions = relationship("ModelPrediction", back_populates="stock", cascade="all, delete-orphan")


# ── price_history ─────────────────────────────────────────────────
class PriceHistory(Base):
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    stock = relationship("Stock", back_populates="price_history")

    __table_args__ = (
        Index("ix_price_history_stock_date", "stock_id", "date"),
        Index("ix_price_history_date", "date"),
    )


# ── backtest_runs ─────────────────────────────────────────────────
class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    strategy = Column(String(20), nullable=False)  # ema / rsi / macd
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)

    # Metrics
    total_return = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    volatility = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    stock = relationship("Stock", back_populates="backtest_runs")
    trades = relationship("Trade", back_populates="backtest_run", cascade="all, delete-orphan")
    portfolio_snapshots = relationship("PortfolioSnapshot", back_populates="backtest_run", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_backtest_runs_stock_strategy", "stock_id", "strategy"),
        Index("ix_backtest_runs_created", "created_at"),
    )


# ── trades ──────────────────────────────────────────────────────────
class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    type = Column(String(10), nullable=False)  # BUY / SELL
    price = Column(Float, nullable=False)
    shares = Column(Integer, nullable=False)
    order_value = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)  # only populated for SELL trades
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    backtest_run = relationship("BacktestRun", back_populates="trades")
    stock = relationship("Stock", back_populates="trades")

    __table_args__ = (
        Index("ix_trades_backtest_date", "backtest_run_id", "date"),
        Index("ix_trades_stock_date", "stock_id", "date"),
    )


# ── portfolio_snapshots ───────────────────────────────────────────
class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    strategy_value = Column(Float, nullable=False)
    benchmark_value = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    backtest_run = relationship("BacktestRun", back_populates="portfolio_snapshots")

    __table_args__ = (
        Index("ix_portfolio_backtest_date", "backtest_run_id", "date"),
    )


# ── model_predictions ─────────────────────────────────────────────
class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    model_type = Column(String(20), nullable=False)  # LSTM / XGBoost
    predicted_value = Column(Float, nullable=True)   # for LSTM: predicted price
    predicted_signal = Column(String(20), nullable=True)  # for XGBoost: BUY / SELL / HOLD
    actual_price = Column(Float, nullable=True)  # filled in later for accuracy tracking
    prediction_date = Column(Date, nullable=False)
    horizon_days = Column(Integer, default=1)
    accuracy = Column(Float, nullable=True)  # computed later: |predicted - actual| / actual
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    stock = relationship("Stock", back_populates="model_predictions")

    __table_args__ = (
        Index("ix_predictions_stock_date", "stock_id", "prediction_date"),
        Index("ix_predictions_model", "model_type", "prediction_date"),
    )
