"""Tests for strategy signal generation and its interaction with the engine."""

import pandas as pd
import pytest

from backend.backtester.engine import BacktestEngine
from backend.strategies.base import (
    EMACrossoverStrategy,
    MACDStrategy,
    RSIMeanReversionStrategy,
)

# Flat, rally, sell-off, rally: forces bullish -> bearish -> bullish crossovers.
CLOSES = [100] * 5 + [110, 120, 130, 120, 100, 80, 70, 80, 100, 120, 140, 150]


def _prices(closes=CLOSES):
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=index)


@pytest.mark.parametrize(
    "strategy",
    [EMACrossoverStrategy(fast=2, slow=4), MACDStrategy(fast=2, slow=4, signal=2),
     RSIMeanReversionStrategy(window=3)],
)
def test_positions_are_only_minus_one_zero_or_one(strategy):
    signals = strategy.generate_signals(_prices())
    assert set(signals["position"].unique()) <= {-1, 0, 1}
    assert signals["position"].isna().sum() == 0


def test_direct_signal_flip_emits_single_step_positions():
    signals = EMACrossoverStrategy(fast=2, slow=4).generate_signals(_prices())
    flips = signals["signal"].diff().abs() == 2
    assert flips.any(), "fixture should contain a direct -1 <-> 1 crossover"
    assert (signals.loc[flips, "position"].abs() == 1).all()


def test_ema_crossover_round_trip_trades_in_engine():
    df = _prices()
    signals = EMACrossoverStrategy(fast=2, slow=4).generate_signals(df)
    trades = BacktestEngine().run(df, signals)["trades"]
    # Buy on the first rally, sell on the bearish cross, re-buy on the next rally.
    assert list(trades["type"]) == ["BUY", "SELL", "BUY"]


def test_generate_signals_does_not_mutate_input():
    df = _prices()
    EMACrossoverStrategy(fast=2, slow=4).generate_signals(df)
    assert list(df.columns) == ["Close"]
