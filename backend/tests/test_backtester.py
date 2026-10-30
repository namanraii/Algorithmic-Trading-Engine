"""Tests for the backtesting engine and its performance metrics."""

import numpy as np
import pandas as pd
import pytest

from backend.backtester.engine import BacktestEngine
from backend.backtester.metrics import MetricsCalculator


def _prices(closes):
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=index)


def _signals(df, positions):
    return pd.DataFrame({"position": positions}, index=df.index)


# --- MetricsCalculator -------------------------------------------------------


def test_metrics_empty_when_fewer_than_two_values():
    calc = MetricsCalculator()
    result = calc.calculate_all(pd.Series([100.0]), pd.DataFrame())
    assert result == calc._empty_metrics()
    assert all(v == 0 for v in result.values())


def test_total_return_and_max_drawdown():
    values = pd.Series([100.0, 120.0, 90.0, 110.0])
    result = MetricsCalculator().calculate_all(values, pd.DataFrame())
    assert result["total_return"] == pytest.approx(10.0)
    # Peak 120 -> trough 90 is a 25% drawdown
    assert result["max_drawdown"] == pytest.approx(-25.0)
    assert result["win_rate"] == 0
    assert result["total_trades"] == 0


def test_flat_portfolio_has_zero_sharpe_and_volatility():
    values = pd.Series([100.0] * 5)
    result = MetricsCalculator().calculate_all(values, pd.DataFrame())
    assert result["sharpe_ratio"] == 0
    assert result["volatility"] == 0
    assert result["max_drawdown"] == 0
    assert result["calmar_ratio"] == 0


def test_sharpe_matches_annualized_formula():
    values = pd.Series([100.0, 101.0, 100.5, 102.0, 103.5])
    rfr = 0.065
    returns = values.pct_change().dropna()
    excess = returns - rfr / 252
    expected = np.sqrt(252) * excess.mean() / excess.std()
    result = MetricsCalculator(risk_free_rate=rfr).calculate_all(values, pd.DataFrame())
    assert result["sharpe_ratio"] == pytest.approx(round(expected, 3))


def test_win_rate_pairs_buys_with_sells():
    trades = pd.DataFrame(
        [
            {"type": "BUY", "price": 100.0},
            {"type": "SELL", "price": 110.0},  # win
            {"type": "BUY", "price": 105.0},
            {"type": "SELL", "price": 95.0},  # loss
            {"type": "BUY", "price": 90.0},  # still open, not counted
        ]
    )
    values = pd.Series([100.0, 101.0, 102.0])
    result = MetricsCalculator().calculate_all(values, trades)
    assert result["win_rate"] == pytest.approx(50.0)
    assert result["total_trades"] == 3


# --- BacktestEngine ----------------------------------------------------------


def test_buy_then_sell_round_trip_applies_commission():
    df = _prices([100.0, 110.0, 120.0])
    signals = _signals(df, [1, 0, -1])
    engine = BacktestEngine(initial_capital=10_000, commission=0.001)

    result = engine.run(df, signals)
    trades = result["trades"]

    shares = int(10_000 * 0.95 / 100.0)  # 95 shares
    cash_after_buy = 10_000 - shares * 100.0 * 1.001
    expected_final = cash_after_buy + shares * 120.0 * 0.999

    assert list(trades["type"]) == ["BUY", "SELL"]
    assert list(trades["shares"]) == [shares, shares]
    assert result["final_capital"] == pytest.approx(expected_final)
    assert result["portfolio"]["value"].iloc[-1] == pytest.approx(expected_final)


def test_sell_signal_without_position_is_ignored():
    df = _prices([100.0, 90.0, 80.0])
    signals = _signals(df, [-1, -1, 0])
    result = BacktestEngine(initial_capital=5_000).run(df, signals)

    assert result["trades"].empty
    assert result["final_capital"] == pytest.approx(5_000)
    assert (result["portfolio"]["value"] == 5_000).all()


def test_benchmark_is_buy_and_hold_from_first_close():
    df = _prices([50.0, 55.0, 60.0])
    signals = _signals(df, [0, 0, 0])
    engine = BacktestEngine(initial_capital=1_000, commission=0.001)
    result = engine.run(df, signals)

    shares = int(1_000 * 0.95 / 50.0)  # 19 shares
    cash = 1_000 - shares * 50.0 * 1.001
    expected = [cash + shares * p for p in (50.0, 55.0, 60.0)]
    assert list(result["portfolio"]["benchmark"]) == pytest.approx(expected)
