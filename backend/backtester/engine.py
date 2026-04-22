import pandas as pd

class BacktestEngine:
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> dict:
        capital = self.initial_capital
        position = 0
        portfolio = []
        trades = []
        
        for i, (date, row) in enumerate(signals.iterrows()):
            price = df.loc[date, 'Close']
            signal = row['position']
            
            if signal == 1 and position == 0:  # Buy
                shares = int(capital * 0.95 / price)
                cost = shares * price * (1 + self.commission)
                capital -= cost
                position = shares
                trades.append({'date': date, 'type': 'BUY', 'price': price, 'shares': shares})
            
            elif signal == -1 and position > 0:  # Sell
                revenue = position * price * (1 - self.commission)
                capital += revenue
                trades.append({'date': date, 'type': 'SELL', 'price': price, 'shares': position})
                position = 0
            
            portfolio_value = capital + position * price
            portfolio.append({'date': date, 'value': portfolio_value})
        
        # Calculate benchmark (Buy and Hold)
        benchmark_portfolio = []
        if not df.empty:
            initial_price = df.iloc[0]['Close']
            benchmark_shares = int(self.initial_capital * 0.95 / initial_price)
            benchmark_cash = self.initial_capital - (benchmark_shares * initial_price * (1 + self.commission))
            
            for date, row in df.iterrows():
                benchmark_portfolio.append({
                    'date': date,
                    'benchmark': benchmark_cash + benchmark_shares * row['Close']
                })
        
        portfolio_df = pd.DataFrame(portfolio).set_index('date')
        benchmark_df = pd.DataFrame(benchmark_portfolio).set_index('date')
        
        # Combine portfolio and benchmark
        if not portfolio_df.empty and not benchmark_df.empty:
            portfolio_df = portfolio_df.join(benchmark_df)
        
        return {
            'portfolio': portfolio_df,
            'trades': pd.DataFrame(trades),
            'final_capital': capital + position * df['Close'].iloc[-1] if not df.empty else self.initial_capital
        }
