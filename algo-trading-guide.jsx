import { useState } from "react";

const phases = [
  {
    id: 1,
    week: "Week 1",
    title: "Foundation & Data Pipeline",
    color: "#00D4AA",
    icon: "⬡",
    days: "Days 1–7",
    goal: "Get your environment, data, and project structure ready. Everything else depends on this being solid.",
    tasks: [
      {
        title: "Project Setup & Environment",
        detail: `Create a clean monorepo structure:\n\nalgo-trading-engine/\n├── backend/\n│   ├── data/          ← raw + processed data\n│   ├── strategies/    ← trading strategy modules\n│   ├── backtester/    ← core backtesting engine\n│   ├── ml_models/     ← LSTM + XGBoost models\n│   ├── api/           ← FastAPI routes\n│   └── utils/         ← helpers, indicators\n├── frontend/          ← React + Recharts dashboard\n├── notebooks/         ← EDA and model experiments\n├── tests/\n└── README.md`,
        code: `# Create virtual environment\npython -m venv venv\nsource venv/bin/activate  # Mac/Linux\n\n# Install core dependencies\npip install yfinance pandas numpy ta fastapi uvicorn\npip install scikit-learn xgboost lightgbm\npip install tensorflow keras  # for LSTM\npip install shap plotly backtrader\npip install pytest python-dotenv`
      },
      {
        title: "Data Ingestion Layer",
        detail: "Build a DataLoader class that fetches OHLCV data (Open, High, Low, Close, Volume) from yfinance. Support multiple tickers, date ranges, and intervals (1d, 1h). Cache data locally to avoid repeated API hits.",
        code: `import yfinance as yf\nimport pandas as pd\nfrom pathlib import Path\n\nclass DataLoader:\n    def __init__(self, cache_dir="data/raw"):\n        self.cache_dir = Path(cache_dir)\n        self.cache_dir.mkdir(parents=True, exist_ok=True)\n\n    def fetch(self, ticker: str, start: str, end: str, interval="1d") -> pd.DataFrame:\n        cache_path = self.cache_dir / f"{ticker}_{start}_{end}_{interval}.csv"\n        if cache_path.exists():\n            return pd.read_csv(cache_path, index_col=0, parse_dates=True)\n        \n        df = yf.download(ticker, start=start, end=end, interval=interval)\n        df.to_csv(cache_path)\n        return df\n\n    def fetch_multiple(self, tickers: list, **kwargs) -> dict:\n        return {t: self.fetch(t, **kwargs) for t in tickers}`
      },
      {
        title: "Technical Indicators Module",
        detail: "Implement a FeatureEngineer class using the `ta` library (Technical Analysis). These indicators become both your strategy signals AND your ML model features.",
        code: `import ta\n\nclass FeatureEngineer:\n    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:\n        # Trend indicators\n        df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)\n        df['ema_50'] = ta.trend.ema_indicator(df['Close'], window=50)\n        df['macd'] = ta.trend.macd(df['Close'])\n        df['macd_signal'] = ta.trend.macd_signal(df['Close'])\n        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])\n        \n        # Momentum\n        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)\n        df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])\n        \n        # Volatility\n        df['bb_upper'] = ta.volatility.bollinger_hband(df['Close'])\n        df['bb_lower'] = ta.volatility.bollinger_lband(df['Close'])\n        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])\n        \n        # Volume\n        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])\n        df['vwap'] = ta.volume.volume_weighted_average_price(\n            df['High'], df['Low'], df['Close'], df['Volume'])\n        \n        return df.dropna()`
      }
    ]
  },
  {
    id: 2,
    week: "Week 2",
    title: "Strategies & Backtesting Engine",
    color: "#FF6B35",
    icon: "◈",
    days: "Days 8–14",
    goal: "Build 3 classic trading strategies and a robust backtesting engine that simulates real trading conditions.",
    tasks: [
      {
        title: "Base Strategy Architecture",
        detail: "Build an abstract Strategy class so all strategies follow the same interface. Each strategy must implement generate_signals() returning a DataFrame with a 'signal' column: 1 = Buy, -1 = Sell, 0 = Hold.",
        code: `from abc import ABC, abstractmethod\nimport pandas as pd\n\nclass BaseStrategy(ABC):\n    def __init__(self, name: str):\n        self.name = name\n    \n    @abstractmethod\n    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:\n        pass\n    \n    def _clean_signals(self, df):\n        df['position'] = df['signal'].diff()\n        df['position'] = df['position'].fillna(0)\n        return df`
      },
      {
        title: "Strategy 1: EMA Crossover",
        detail: "Golden Cross / Death Cross — when the 20-day EMA crosses above the 50-day EMA, go long. When it crosses below, go short. This is one of the most well-known trend-following strategies in the industry.",
        code: `class EMACrossoverStrategy(BaseStrategy):\n    def __init__(self, fast=20, slow=50):\n        super().__init__("EMA Crossover")\n        self.fast = fast\n        self.slow = slow\n    \n    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:\n        df = df.copy()\n        df['ema_fast'] = df['Close'].ewm(span=self.fast).mean()\n        df['ema_slow'] = df['Close'].ewm(span=self.slow).mean()\n        \n        df['signal'] = 0\n        df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] = 1\n        df.loc[df['ema_fast'] < df['ema_slow'], 'signal'] = -1\n        \n        return self._clean_signals(df)`
      },
      {
        title: "Strategy 2: RSI Mean Reversion",
        detail: "When RSI drops below 30 (oversold), expect price to bounce → Buy. When RSI rises above 70 (overbought), expect a pullback → Sell. Classic mean reversion. Pair with Bollinger Bands for confirmation.",
        code: `class RSIMeanReversionStrategy(BaseStrategy):\n    def __init__(self, oversold=30, overbought=70, window=14):\n        super().__init__("RSI Mean Reversion")\n        self.oversold = oversold\n        self.overbought = overbought\n        self.window = window\n    \n    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:\n        df = df.copy()\n        delta = df['Close'].diff()\n        gain = delta.clip(lower=0).rolling(self.window).mean()\n        loss = -delta.clip(upper=0).rolling(self.window).mean()\n        rs = gain / loss\n        df['rsi'] = 100 - (100 / (1 + rs))\n        \n        df['signal'] = 0\n        df.loc[df['rsi'] < self.oversold, 'signal'] = 1\n        df.loc[df['rsi'] > self.overbought, 'signal'] = -1\n        \n        return self._clean_signals(df)`
      },
      {
        title: "Strategy 3: MACD Momentum",
        detail: "MACD (Moving Average Convergence Divergence) captures momentum. When MACD line crosses above the signal line → Buy. When it crosses below → Sell. The histogram shows strength of the trend.",
        code: `class MACDStrategy(BaseStrategy):\n    def __init__(self, fast=12, slow=26, signal=9):\n        super().__init__("MACD Momentum")\n        self.fast = fast\n        self.slow = slow\n        self.signal_period = signal\n    \n    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:\n        df = df.copy()\n        ema_fast = df['Close'].ewm(span=self.fast).mean()\n        ema_slow = df['Close'].ewm(span=self.slow).mean()\n        df['macd'] = ema_fast - ema_slow\n        df['macd_signal'] = df['macd'].ewm(span=self.signal_period).mean()\n        df['macd_hist'] = df['macd'] - df['macd_signal']\n        \n        df['signal'] = 0\n        df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1\n        df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1\n        \n        return self._clean_signals(df)`
      },
      {
        title: "Backtesting Engine",
        detail: "The core engine: simulates trading with realistic constraints — transaction costs (0.1% per trade), slippage, position sizing, and tracks portfolio value over time. This is what separates a toy project from a real one.",
        code: `class BacktestEngine:\n    def __init__(self, initial_capital=100000, commission=0.001):\n        self.initial_capital = initial_capital\n        self.commission = commission\n    \n    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> dict:\n        capital = self.initial_capital\n        position = 0\n        portfolio = []\n        trades = []\n        \n        for i, (date, row) in enumerate(signals.iterrows()):\n            price = df.loc[date, 'Close']\n            signal = row['position']\n            \n            if signal == 1 and position == 0:  # Buy\n                shares = int(capital * 0.95 / price)\n                cost = shares * price * (1 + self.commission)\n                capital -= cost\n                position = shares\n                trades.append({'date': date, 'type': 'BUY', 'price': price, 'shares': shares})\n            \n            elif signal == -1 and position > 0:  # Sell\n                revenue = position * price * (1 - self.commission)\n                capital += revenue\n                trades.append({'date': date, 'type': 'SELL', 'price': price, 'shares': position})\n                position = 0\n            \n            portfolio_value = capital + position * price\n            portfolio.append({'date': date, 'value': portfolio_value})\n        \n        return {\n            'portfolio': pd.DataFrame(portfolio).set_index('date'),\n            'trades': pd.DataFrame(trades),\n            'final_capital': capital + position * df['Close'].iloc[-1]\n        }`
      }
    ]
  },
  {
    id: 3,
    week: "Week 2–3",
    title: "Performance Metrics",
    color: "#7B61FF",
    icon: "◇",
    days: "Days 12–16",
    goal: "Calculate every metric a quant finance interviewer would expect. This is what makes your project credible.",
    tasks: [
      {
        title: "Core Metrics Calculator",
        detail: "These 7 metrics are the standard vocabulary of quantitative finance. Know what each one means and be able to explain them in an interview — especially Sharpe Ratio, Max Drawdown, and Calmar Ratio.",
        code: `import numpy as np\n\nclass MetricsCalculator:\n    def __init__(self, risk_free_rate=0.065):  # 6.5% India RBI rate\n        self.rfr = risk_free_rate\n    \n    def calculate_all(self, portfolio_values: pd.Series, trades: pd.DataFrame) -> dict:\n        returns = portfolio_values.pct_change().dropna()\n        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100\n        \n        # Annualized Sharpe Ratio — risk-adjusted return\n        excess_returns = returns - self.rfr / 252\n        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()\n        \n        # Max Drawdown — worst peak-to-trough loss\n        rolling_max = portfolio_values.cummax()\n        drawdown = (portfolio_values - rolling_max) / rolling_max\n        max_drawdown = drawdown.min() * 100\n        \n        # Calmar Ratio — return per unit of max drawdown\n        annual_return = total_return * (252 / len(returns))\n        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0\n        \n        # Win Rate\n        if len(trades) > 1:\n            sell_trades = trades[trades['type'] == 'SELL']\n            buy_trades = trades[trades['type'] == 'BUY']\n            pnl = sell_trades['price'].values - buy_trades['price'].values[:len(sell_trades)]\n            win_rate = (pnl > 0).sum() / len(pnl) * 100\n        else:\n            win_rate = 0\n        \n        return {\n            'total_return': round(total_return, 2),\n            'sharpe_ratio': round(sharpe, 3),\n            'max_drawdown': round(max_drawdown, 2),\n            'calmar_ratio': round(calmar, 3),\n            'win_rate': round(win_rate, 2),\n            'total_trades': len(trades[trades['type'] == 'BUY']) if len(trades) else 0,\n            'volatility': round(returns.std() * np.sqrt(252) * 100, 2)\n        }`
      }
    ]
  },
  {
    id: 4,
    week: "Week 3",
    title: "ML Forecasting Layer",
    color: "#FFD93D",
    icon: "⬟",
    days: "Days 17–23",
    goal: "Add LSTM for price prediction and XGBoost for signal classification. This is the AI/ML layer that makes this project unique vs. standard algo trading repos.",
    tasks: [
      {
        title: "LSTM Price Forecaster",
        detail: "LSTM (Long Short-Term Memory) networks are ideal for time series because they remember long-term patterns. Train it to predict the next day's closing price using the last 60 days of OHLCV + indicator data.",
        code: `import numpy as np\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import LSTM, Dense, Dropout\nfrom sklearn.preprocessing import MinMaxScaler\n\nclass LSTMForecaster:\n    def __init__(self, sequence_length=60):\n        self.seq_len = sequence_length\n        self.scaler = MinMaxScaler()\n        self.model = None\n    \n    def _build_model(self, n_features):\n        model = Sequential([\n            LSTM(128, return_sequences=True, input_shape=(self.seq_len, n_features)),\n            Dropout(0.2),\n            LSTM(64, return_sequences=False),\n            Dropout(0.2),\n            Dense(32, activation='relu'),\n            Dense(1)  # predict next close price\n        ])\n        model.compile(optimizer='adam', loss='huber')  # huber is robust to outliers\n        return model\n    \n    def prepare_sequences(self, df: pd.DataFrame, feature_cols: list):\n        data = self.scaler.fit_transform(df[feature_cols + ['Close']].values)\n        X, y = [], []\n        for i in range(self.seq_len, len(data)):\n            X.append(data[i-self.seq_len:i, :-1])  # features\n            y.append(data[i, -1])                   # close price\n        return np.array(X), np.array(y)\n    \n    def train(self, df, feature_cols, epochs=50, validation_split=0.15):\n        X, y = self.prepare_sequences(df, feature_cols)\n        split = int(len(X) * 0.8)\n        self.model = self._build_model(len(feature_cols))\n        history = self.model.fit(\n            X[:split], y[:split],\n            validation_data=(X[split:], y[split:]),\n            epochs=epochs, batch_size=32, verbose=1\n        )\n        return history`
      },
      {
        title: "XGBoost Signal Classifier",
        detail: "Instead of predicting price, XGBoost classifies the signal: will the stock go up >1% tomorrow? This is a classification problem (0 = no, 1 = yes). Use technical indicators as features. XGBoost typically outperforms LSTM on tabular data.",
        code: `import xgboost as xgb\nfrom sklearn.model_selection import TimeSeriesSplit\nfrom sklearn.metrics import classification_report\nimport shap\n\nclass XGBoostSignalClassifier:\n    def __init__(self, threshold=0.01):\n        self.threshold = threshold  # 1% price move = positive class\n        self.model = None\n    \n    def prepare_labels(self, df: pd.DataFrame) -> pd.Series:\n        future_return = df['Close'].shift(-1) / df['Close'] - 1\n        return (future_return > self.threshold).astype(int)\n    \n    def train(self, df: pd.DataFrame, feature_cols: list):\n        X = df[feature_cols].dropna()\n        y = self.prepare_labels(df).loc[X.index]\n        \n        # TimeSeriesSplit — critical: no lookahead bias!\n        tscv = TimeSeriesSplit(n_splits=5)\n        \n        self.model = xgb.XGBClassifier(\n            n_estimators=500, learning_rate=0.05,\n            max_depth=6, subsample=0.8,\n            colsample_bytree=0.8, eval_metric='logloss'\n        )\n        \n        for train_idx, val_idx in tscv.split(X):\n            self.model.fit(\n                X.iloc[train_idx], y.iloc[train_idx],\n                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],\n                verbose=False\n            )\n        \n        return classification_report(y, self.model.predict(X))\n    \n    def explain(self, X):  # SHAP explainability\n        explainer = shap.TreeExplainer(self.model)\n        return explainer.shap_values(X)`
      }
    ]
  },
  {
    id: 5,
    week: "Week 3–4",
    title: "FastAPI Backend",
    color: "#00B4D8",
    icon: "▣",
    days: "Days 22–26",
    goal: "Expose all your backtesting and ML capabilities through a clean REST API that the React frontend will consume.",
    tasks: [
      {
        title: "API Routes",
        detail: "Build 4 core endpoints. The frontend will call these to run backtests, compare strategies, get ML predictions, and fetch performance metrics. Keep responses clean and JSON-serializable.",
        code: `from fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI(title="Algo Trading Engine")\napp.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])\n\nclass BacktestRequest(BaseModel):\n    ticker: str\n    strategy: str  # 'ema', 'rsi', 'macd', 'ml'\n    start_date: str\n    end_date: str\n    initial_capital: float = 100000\n\n@app.post("/api/backtest")\nasync def run_backtest(req: BacktestRequest):\n    loader = DataLoader()\n    df = loader.fetch(req.ticker, req.start_date, req.end_date)\n    \n    fe = FeatureEngineer()\n    df = fe.add_all_indicators(df)\n    \n    strategy_map = {\n        'ema': EMACrossoverStrategy(),\n        'rsi': RSIMeanReversionStrategy(),\n        'macd': MACDStrategy()\n    }\n    \n    strategy = strategy_map[req.strategy]\n    signals = strategy.generate_signals(df)\n    engine = BacktestEngine(initial_capital=req.initial_capital)\n    result = engine.run(df, signals)\n    \n    metrics = MetricsCalculator().calculate_all(\n        result['portfolio']['value'], result['trades']\n    )\n    \n    return {\n        'portfolio': result['portfolio'].reset_index().to_dict('records'),\n        'trades': result['trades'].to_dict('records'),\n        'metrics': metrics\n    }\n\n@app.post("/api/compare")\nasync def compare_strategies(ticker: str, start: str, end: str):\n    # Run all 3 strategies and return metrics side by side\n    ...\n\n@app.get("/api/predict/{ticker}")\nasync def get_ml_prediction(ticker: str):\n    # Return LSTM price forecast + XGBoost signal\n    ...`
      }
    ]
  },
  {
    id: 6,
    week: "Week 4",
    title: "React Dashboard",
    color: "#FF4D6D",
    icon: "◉",
    days: "Days 26–30",
    goal: "Build a professional trading dashboard. This is what recruiters and interviewers will actually see — make it impressive.",
    tasks: [
      {
        title: "Dashboard Components",
        detail: "Build 5 core components using React + Recharts. The dashboard should feel like a real trading terminal — dark theme, live-looking charts, clean data tables.",
        code: `// Key components to build:\n// 1. TickerSelector — search and select stocks (AAPL, GOOGL, INFY.NS, RELIANCE.NS)\n// 2. BacktestControls — date range picker, strategy selector, capital input\n// 3. PortfolioChart — area chart of portfolio value over time + benchmark (Buy & Hold)\n// 4. MetricsCards — Sharpe, Drawdown, Win Rate, Total Return in stat cards\n// 5. TradeLog — table of all buy/sell trades with PnL per trade\n// 6. StrategyComparison — bar chart comparing all 3 strategies side-by-side\n\n// Recharts example for portfolio chart:\nimport { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';\n\nconst PortfolioChart = ({ data, benchmarkData }) => (\n  <ResponsiveContainer width="100%" height={350}>\n    <AreaChart data={data}>\n      <defs>\n        <linearGradient id="stratGrad" x1="0" y1="0" x2="0" y2="1">\n          <stop offset="5%" stopColor="#00D4AA" stopOpacity={0.3}/>\n          <stop offset="95%" stopColor="#00D4AA" stopOpacity={0}/>\n        </linearGradient>\n      </defs>\n      <XAxis dataKey="date" tick={{ fill: '#888', fontSize: 11 }} />\n      <YAxis tick={{ fill: '#888', fontSize: 11 }} />\n      <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #333' }} />\n      <Area type="monotone" dataKey="value" stroke="#00D4AA"\n            fill="url(#stratGrad)" name="Strategy" />\n      <Area type="monotone" dataKey="benchmark" stroke="#FF6B35"\n            fill="none" strokeDasharray="4 4" name="Buy & Hold" />\n    </AreaChart>\n  </ResponsiveContainer>\n);`
      },
      {
        title: "Deployment",
        detail: "Deploy backend to Render (free tier, same as NovaMentor). Deploy frontend to Vercel. Add environment variables for API URL. Update README with live demo link — this is critical for your resume.",
        code: `# Backend — Render deployment\n# 1. Push to GitHub\n# 2. Connect repo on render.com\n# 3. Build command: pip install -r requirements.txt\n# 4. Start command: uvicorn api.main:app --host 0.0.0.0 --port $PORT\n\n# Frontend — Vercel deployment\nnpx vercel\n# Set environment variable:\n# VITE_API_URL = https://your-backend.onrender.com\n\n# In React, use:\nconst API = import.meta.env.VITE_API_URL;\nconst response = await fetch(\`\${API}/api/backtest\`, {...});`
      }
    ]
  }
];

const metrics = [
  { name: "Sharpe Ratio", formula: "(Rp - Rf) / σp", explain: "Return per unit of risk. >1 is good, >2 is excellent. Always annualized." },
  { name: "Max Drawdown", formula: "(Trough - Peak) / Peak", explain: "Worst peak-to-trough loss. -15% means you'd have lost 15% at the worst point." },
  { name: "Calmar Ratio", formula: "Annual Return / |Max Drawdown|", explain: "How much return per unit of drawdown risk. Higher = better." },
  { name: "Win Rate", formula: "Winning Trades / Total Trades", explain: "% of trades that were profitable. 50%+ is good for trend strategies." },
  { name: "Volatility", formula: "σ(daily returns) × √252", explain: "Annualized standard deviation of returns. Lower = more stable." },
];

const interviewQs = [
  { q: "What is lookahead bias and how did you prevent it?", a: "Using TimeSeriesSplit instead of random train/test split. Features are computed only from past data." },
  { q: "Why use Sharpe Ratio over just total return?", a: "Total return ignores risk taken. A 30% return with 50% volatility is worse than a 20% return with 5% volatility." },
  { q: "Why did you choose LSTM over simpler models?", a: "LSTM captures temporal dependencies and long-range patterns in price sequences that ARIMA/linear models miss." },
  { q: "What is survivorship bias in backtesting?", a: "Testing only on stocks that exist today ignores companies that went bankrupt. Use point-in-time data if possible." },
  { q: "How is your ML strategy different from rule-based strategies?", a: "ML learns non-linear relationships across 15+ indicators simultaneously, whereas rules are hand-crafted thresholds." },
];

const techStack = [
  { layer: "Data", items: ["yfinance", "pandas", "numpy", "ta (Technical Analysis)"] },
  { layer: "ML/DL", items: ["TensorFlow/Keras (LSTM)", "XGBoost", "scikit-learn", "SHAP"] },
  { layer: "Backend", items: ["FastAPI", "Pydantic", "Uvicorn", "pytest"] },
  { layer: "Frontend", items: ["React + Vite", "Recharts", "Tailwind CSS", "Axios"] },
  { layer: "Deploy", items: ["Render (backend)", "Vercel (frontend)", "GitHub Actions (CI)"] },
];

export default function AlgoTradingGuide() {
  const [activePhase, setActivePhase] = useState(1);
  const [activeTask, setActiveTask] = useState(0);
  const [activeTab, setActiveTab] = useState("roadmap");
  const [showCode, setShowCode] = useState({});

  const currentPhase = phases.find(p => p.id === activePhase);

  const toggleCode = (key) => setShowCode(prev => ({ ...prev, [key]: !prev[key] }));

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      background: "#0a0a0f",
      minHeight: "100vh",
      color: "#e0e0e0",
      padding: "0",
    }}>
      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #0d0d1a 0%, #0a0a0f 100%)",
        borderBottom: "1px solid #1a1a2e",
        padding: "28px 40px",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 6 }}>
            <div style={{
              background: "linear-gradient(135deg, #00D4AA, #7B61FF)",
              borderRadius: 8,
              width: 36, height: 36,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 18, fontWeight: 900
            }}>⟨/⟩</div>
            <div>
              <div style={{ fontSize: 11, color: "#00D4AA", letterSpacing: 3, textTransform: "uppercase" }}>Portfolio Project — Fintech</div>
              <div style={{ fontSize: 22, fontWeight: 800, color: "#fff", letterSpacing: -0.5 }}>Algorithmic Trading Engine</div>
            </div>
          </div>
          <div style={{ fontSize: 12, color: "#555", marginLeft: 52 }}>4-week build plan · Python + FastAPI + React · ML-powered backtesting</div>
        </div>
      </div>

      {/* Nav Tabs */}
      <div style={{
        borderBottom: "1px solid #1a1a2e",
        background: "#0a0a0f",
        padding: "0 40px",
        position: "sticky",
        top: 88,
        zIndex: 99,
      }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", gap: 0 }}>
          {[
            { id: "roadmap", label: "Build Roadmap" },
            { id: "metrics", label: "Finance Metrics" },
            { id: "stack", label: "Tech Stack" },
            { id: "interview", label: "Interview Prep" },
          ].map(tab => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
              background: "none",
              border: "none",
              borderBottom: activeTab === tab.id ? "2px solid #00D4AA" : "2px solid transparent",
              color: activeTab === tab.id ? "#00D4AA" : "#555",
              padding: "14px 20px",
              cursor: "pointer",
              fontSize: 12,
              fontFamily: "inherit",
              letterSpacing: 1,
              textTransform: "uppercase",
              transition: "all 0.2s",
              fontWeight: activeTab === tab.id ? 700 : 400,
            }}>{tab.label}</button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 40px" }}>

        {/* ROADMAP TAB */}
        {activeTab === "roadmap" && (
          <div>
            {/* Phase selector */}
            <div style={{ display: "flex", gap: 12, marginBottom: 32, flexWrap: "wrap" }}>
              {phases.map(phase => (
                <button key={phase.id} onClick={() => { setActivePhase(phase.id); setActiveTask(0); }} style={{
                  background: activePhase === phase.id ? phase.color + "15" : "transparent",
                  border: `1px solid ${activePhase === phase.id ? phase.color : "#1a1a2e"}`,
                  borderRadius: 8,
                  padding: "10px 18px",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  color: activePhase === phase.id ? phase.color : "#555",
                  fontSize: 11,
                  letterSpacing: 0.5,
                  transition: "all 0.2s",
                  textAlign: "left",
                }}>
                  <div style={{ fontWeight: 700, marginBottom: 2 }}>{phase.icon} {phase.week}</div>
                  <div style={{ fontSize: 10, opacity: 0.8 }}>{phase.title}</div>
                </button>
              ))}
            </div>

            {/* Phase detail */}
            {currentPhase && (
              <div>
                <div style={{
                  background: "#0d0d1a",
                  border: `1px solid ${currentPhase.color}30`,
                  borderLeft: `3px solid ${currentPhase.color}`,
                  borderRadius: 10,
                  padding: "20px 24px",
                  marginBottom: 24,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div>
                      <div style={{ fontSize: 10, color: currentPhase.color, letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>{currentPhase.days}</div>
                      <div style={{ fontSize: 20, fontWeight: 800, color: "#fff", marginBottom: 8 }}>{currentPhase.title}</div>
                      <div style={{ fontSize: 12, color: "#888", lineHeight: 1.6 }}>{currentPhase.goal}</div>
                    </div>
                    <div style={{
                      fontSize: 36,
                      color: currentPhase.color,
                      opacity: 0.4,
                      marginLeft: 24,
                    }}>{currentPhase.icon}</div>
                  </div>
                </div>

                {/* Task tabs */}
                <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
                  {currentPhase.tasks.map((task, i) => (
                    <button key={i} onClick={() => setActiveTask(i)} style={{
                      background: activeTask === i ? currentPhase.color + "20" : "#0d0d1a",
                      border: `1px solid ${activeTask === i ? currentPhase.color : "#1a1a2e"}`,
                      borderRadius: 6,
                      padding: "8px 14px",
                      cursor: "pointer",
                      fontFamily: "inherit",
                      color: activeTask === i ? currentPhase.color : "#555",
                      fontSize: 10,
                      letterSpacing: 0.5,
                      transition: "all 0.2s",
                      maxWidth: 200,
                      textAlign: "left",
                    }}>{task.title}</button>
                  ))}
                </div>

                {/* Task detail */}
                {currentPhase.tasks[activeTask] && (() => {
                  const task = currentPhase.tasks[activeTask];
                  const codeKey = `${currentPhase.id}-${activeTask}`;
                  return (
                    <div style={{
                      background: "#0d0d1a",
                      border: "1px solid #1a1a2e",
                      borderRadius: 10,
                      overflow: "hidden",
                    }}>
                      <div style={{ padding: "20px 24px", borderBottom: "1px solid #1a1a2e" }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", marginBottom: 10 }}>{task.title}</div>
                        <div style={{ fontSize: 12, color: "#888", lineHeight: 1.7, whiteSpace: "pre-line" }}>{task.detail}</div>
                      </div>
                      <div>
                        <button onClick={() => toggleCode(codeKey)} style={{
                          width: "100%",
                          background: showCode[codeKey] ? "#111120" : "#0a0a14",
                          border: "none",
                          borderBottom: showCode[codeKey] ? "1px solid #1a1a2e" : "none",
                          color: currentPhase.color,
                          padding: "10px 24px",
                          textAlign: "left",
                          cursor: "pointer",
                          fontFamily: "inherit",
                          fontSize: 11,
                          letterSpacing: 1,
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                        }}>
                          <span>{showCode[codeKey] ? "▼" : "▶"}</span>
                          <span>{showCode[codeKey] ? "HIDE CODE" : "SHOW CODE"}</span>
                        </button>
                        {showCode[codeKey] && (
                          <pre style={{
                            margin: 0,
                            padding: "20px 24px",
                            background: "#070710",
                            color: "#c9d1d9",
                            fontSize: 11,
                            lineHeight: 1.7,
                            overflowX: "auto",
                            borderTop: "none",
                          }}><code>{task.code}</code></pre>
                        )}
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        )}

        {/* METRICS TAB */}
        {activeTab === "metrics" && (
          <div>
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 11, color: "#00D4AA", letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>Know These Cold</div>
              <div style={{ fontSize: 18, fontWeight: 800, color: "#fff", marginBottom: 4 }}>Performance Metrics</div>
              <div style={{ fontSize: 12, color: "#555" }}>Every metric you must implement and explain in interviews</div>
            </div>
            <div style={{ display: "grid", gap: 16 }}>
              {metrics.map((m, i) => (
                <div key={i} style={{
                  background: "#0d0d1a",
                  border: "1px solid #1a1a2e",
                  borderRadius: 10,
                  padding: "20px 24px",
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr 2fr",
                  gap: 24,
                  alignItems: "center",
                }}>
                  <div>
                    <div style={{ fontSize: 10, color: "#555", marginBottom: 4, letterSpacing: 1 }}>METRIC</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "#fff" }}>{m.name}</div>
                  </div>
                  <div style={{
                    background: "#070710",
                    borderRadius: 6,
                    padding: "8px 12px",
                    border: "1px solid #1a1a2e",
                  }}>
                    <div style={{ fontSize: 10, color: "#555", marginBottom: 4, letterSpacing: 1 }}>FORMULA</div>
                    <div style={{ fontSize: 12, color: "#7B61FF", fontStyle: "italic" }}>{m.formula}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: "#555", marginBottom: 4, letterSpacing: 1 }}>WHAT IT MEANS</div>
                    <div style={{ fontSize: 12, color: "#888", lineHeight: 1.6 }}>{m.explain}</div>
                  </div>
                </div>
              ))}
            </div>

            <div style={{
              marginTop: 28,
              background: "#0d0d1a",
              border: "1px solid #FFD93D30",
              borderLeft: "3px solid #FFD93D",
              borderRadius: 10,
              padding: "20px 24px",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#FFD93D", marginBottom: 12 }}>⚡ Benchmark Comparison</div>
              <div style={{ fontSize: 12, color: "#888", lineHeight: 1.7 }}>
                Always compare your strategy against <span style={{ color: "#fff" }}>Buy & Hold</span> (the simplest benchmark). If your strategy doesn't beat buy-and-hold after transaction costs, it's not useful. Also compare against <span style={{ color: "#fff" }}>NIFTY 50 or S&P 500 index returns</span> for the same period — this is called "alpha generation" and is what every fund manager is evaluated on.
              </div>
            </div>
          </div>
        )}

        {/* TECH STACK TAB */}
        {activeTab === "stack" && (
          <div>
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 11, color: "#00D4AA", letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>What You'll Use</div>
              <div style={{ fontSize: 18, fontWeight: 800, color: "#fff", marginBottom: 4 }}>Full Tech Stack</div>
              <div style={{ fontSize: 12, color: "#555" }}>Every library and tool, with why you're using it</div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
              {techStack.map((layer, i) => {
                const colors = ["#00D4AA", "#FF6B35", "#7B61FF", "#00B4D8", "#FF4D6D"];
                return (
                  <div key={i} style={{
                    background: "#0d0d1a",
                    border: `1px solid ${colors[i]}20`,
                    borderTop: `2px solid ${colors[i]}`,
                    borderRadius: 10,
                    padding: "18px 20px",
                  }}>
                    <div style={{ fontSize: 10, color: colors[i], letterSpacing: 2, textTransform: "uppercase", marginBottom: 14, fontWeight: 700 }}>{layer.layer}</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {layer.items.map((item, j) => (
                        <div key={j} style={{
                          background: "#070710",
                          border: "1px solid #1a1a2e",
                          borderRadius: 6,
                          padding: "6px 12px",
                          fontSize: 12,
                          color: "#ccc",
                          fontFamily: "monospace",
                        }}>{item}</div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* GitHub repo structure */}
            <div style={{
              background: "#070710",
              border: "1px solid #1a1a2e",
              borderRadius: 10,
              padding: "20px 24px",
            }}>
              <div style={{ fontSize: 11, color: "#555", letterSpacing: 2, textTransform: "uppercase", marginBottom: 16 }}>Repo Structure</div>
              <pre style={{ margin: 0, fontSize: 12, color: "#666", lineHeight: 1.8 }}>
{`algo-trading-engine/
├── 📁 backend/
│   ├── data/          ← DataLoader, caching
│   ├── strategies/    ← EMA, RSI, MACD, ML strategies
│   ├── backtester/    ← BacktestEngine + MetricsCalculator
│   ├── ml_models/     ← LSTM + XGBoost + SHAP
│   └── api/           ← FastAPI routes
├── 📁 frontend/
│   ├── src/components/
│   │   ├── Dashboard.jsx
│   │   ├── PortfolioChart.jsx
│   │   ├── MetricsCards.jsx
│   │   ├── TradeLog.jsx
│   │   └── StrategyComparison.jsx
├── 📁 notebooks/      ← EDA.ipynb, ModelExperiments.ipynb
├── 📁 tests/          ← pytest unit tests
├── requirements.txt
└── README.md          ← with live demo GIF + screenshots`}
              </pre>
            </div>
          </div>
        )}

        {/* INTERVIEW PREP TAB */}
        {activeTab === "interview" && (
          <div>
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 11, color: "#00D4AA", letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>Be Ready For These</div>
              <div style={{ fontSize: 18, fontWeight: 800, color: "#fff", marginBottom: 4 }}>Interview Questions</div>
              <div style={{ fontSize: 12, color: "#555" }}>Questions a fintech recruiter will ask about this project</div>
            </div>

            <div style={{ display: "grid", gap: 14 }}>
              {interviewQs.map((qa, i) => (
                <div key={i} style={{
                  background: "#0d0d1a",
                  border: "1px solid #1a1a2e",
                  borderRadius: 10,
                  overflow: "hidden",
                }}>
                  <div style={{ padding: "16px 20px", borderBottom: "1px solid #1a1a2e", display: "flex", gap: 12, alignItems: "flex-start" }}>
                    <div style={{
                      background: "#7B61FF20",
                      border: "1px solid #7B61FF40",
                      borderRadius: 6,
                      width: 24, height: 24,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 11, color: "#7B61FF", fontWeight: 700, flexShrink: 0
                    }}>Q</div>
                    <div style={{ fontSize: 13, color: "#ddd", lineHeight: 1.5, fontWeight: 500 }}>{qa.q}</div>
                  </div>
                  <div style={{ padding: "14px 20px", display: "flex", gap: 12, alignItems: "flex-start" }}>
                    <div style={{
                      background: "#00D4AA20",
                      border: "1px solid #00D4AA40",
                      borderRadius: 6,
                      width: 24, height: 24,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 11, color: "#00D4AA", fontWeight: 700, flexShrink: 0
                    }}>A</div>
                    <div style={{ fontSize: 12, color: "#888", lineHeight: 1.6 }}>{qa.a}</div>
                  </div>
                </div>
              ))}
            </div>

            <div style={{
              marginTop: 28,
              background: "#0d0d1a",
              border: "1px solid #FF6B3530",
              borderLeft: "3px solid #FF6B35",
              borderRadius: 10,
              padding: "20px 24px",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#FF6B35", marginBottom: 12 }}>🎯 Resume Bullet Points</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {[
                  "Built an end-to-end Algorithmic Trading Engine implementing EMA Crossover, RSI Mean Reversion, and MACD Momentum strategies with a custom backtesting engine supporting transaction costs and slippage simulation",
                  "Developed LSTM-based price forecasting model (60-day sequence input) and XGBoost signal classifier with SHAP explainability, achieving X% accuracy on out-of-sample test data",
                  "Computed 7 industry-standard performance metrics (Sharpe Ratio, Max Drawdown, Calmar Ratio, Win Rate, Volatility) enabling quantitative strategy comparison against Buy & Hold benchmark",
                  "Deployed REST API (FastAPI) + React dashboard to Render/Vercel with real-time strategy comparison visualization using Recharts",
                ].map((bullet, i) => (
                  <div key={i} style={{
                    background: "#070710",
                    borderRadius: 6,
                    padding: "10px 14px",
                    fontSize: 11,
                    color: "#888",
                    lineHeight: 1.6,
                    display: "flex",
                    gap: 10,
                    alignItems: "flex-start",
                  }}>
                    <span style={{ color: "#FF6B35", flexShrink: 0 }}>▸</span>
                    <span>{bullet}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
