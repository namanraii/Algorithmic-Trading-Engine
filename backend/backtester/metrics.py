import numpy as np
import pandas as pd

class MetricsCalculator:
    def __init__(self, risk_free_rate=0.065):  # 6.5% India RBI rate as per guide
        self.rfr = risk_free_rate
    
    def calculate_all(self, portfolio_values: pd.Series, trades: pd.DataFrame) -> dict:
        if len(portfolio_values) < 2:
            return self._empty_metrics()

        returns = portfolio_values.pct_change().dropna()
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
        
        # Annualized Sharpe Ratio — risk-adjusted return
        excess_returns = returns - self.rfr / 252
        std_dev = excess_returns.std()
        if std_dev == 0 or pd.isna(std_dev):
            sharpe = 0
        else:
            sharpe = np.sqrt(252) * excess_returns.mean() / std_dev
        
        # Max Drawdown — worst peak-to-trough loss
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Calmar Ratio — return per unit of max drawdown
        annual_return = total_return * (252 / len(returns))
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 and not pd.isna(max_drawdown) else 0
        
        # Win Rate
        if len(trades) > 1:
            sell_trades = trades[trades['type'] == 'SELL']
            buy_trades = trades[trades['type'] == 'BUY']
            # Only count completed trades
            min_len = min(len(sell_trades), len(buy_trades))
            if min_len > 0:
                pnl = sell_trades['price'].values[:min_len] - buy_trades['price'].values[:min_len]
                win_rate = (pnl > 0).sum() / min_len * 100
            else:
                win_rate = 0
        else:
            win_rate = 0
            
        volatility = returns.std() * np.sqrt(252) * 100
        if pd.isna(volatility):
            volatility = 0
        
        return {
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe, 3) if not pd.isna(sharpe) else 0,
            'max_drawdown': round(max_drawdown, 2) if not pd.isna(max_drawdown) else 0,
            'calmar_ratio': round(calmar, 3) if not pd.isna(calmar) else 0,
            'win_rate': round(win_rate, 2),
            'total_trades': len(trades[trades['type'] == 'BUY']) if len(trades) else 0,
            'volatility': round(volatility, 2)
        }

    def _empty_metrics(self):
        return {
            'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
            'calmar_ratio': 0, 'win_rate': 0, 'total_trades': 0, 'volatility': 0
        }
