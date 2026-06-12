# Algorithmic Trading Engine with AI Forecasting 📈

A high-performance, full-stack platform designed for quantitative traders and developers to build, test, and analyze algorithmic trading strategies. Unlike traditional retail trading apps that focus solely on order execution, this engine provides a comprehensive environment for **Strategy Validation**, **AI-Driven Prediction**, and **Persistent Historical Analytics**.

## 🔗 Live Demo
- **Dashboard**: [https://algorithmic-trading-engine.vercel.app/](https://algorithmic-trading-engine.vercel.app/)
- **API Backend**: [https://algorithmic-trading-engine.onrender.com/](https://algorithmic-trading-engine.onrender.com/)

![Dashboard Preview](https://raw.githubusercontent.com/namanraii/Algorithmic-Trading-Engine/main/docs/dashboard_preview.png) *(Placeholder for preview)*

---

## 🚀 Key Features

- **Backtesting Engine**: Simulate complex strategies on years of historical data from Yahoo Finance with 0.1% commission modeling.
- **AI Ensemble Forecasting**:
    - **LSTM (RNN)**: Deep learning model trained on-the-fly for time-series price forecasting.
    - **XGBoost**: Gradient Boosting classifier to identify high-probability BUY/SELL signals.
- **Quant Metrics Suite**: Professional performance analysis including:
    - **Sharpe Ratio** (Risk-adjusted return)
    - **Max Drawdown** (Peak-to-trough loss)
    - **Win Rate** & **Total Trades**
    - **Annualized Volatility** & **Calmar Ratio**
- **Technical Analysis Library**: Integrated `ta` library providing 15+ indicators (EMA, RSI, MACD, Bollinger Bands, ATR, etc.).
- **Interactive Dashboard**: A professional, terminal-inspired React interface featuring real-time equity curves and trade execution logs.
- **Persistent Database Layer**: Postgres-backed storage for price history, backtest runs, trades, portfolio snapshots, and model predictions — enabling historical analytics and faster dashboard loads.

---

## 🏗 Architecture Evolution: Before → After

> **Interview framing**: *"The engine originally computed everything live from yfinance per-request, which meant no historical record of trades or predictions and slow repeated computation. I added a Postgres layer to persist backtest runs and model predictions, which both improved performance (cached results) and enabled analytical queries the original architecture couldn't support."*

### Before (Ephemeral)
- Every API call fetched fresh data from Yahoo Finance
- Backtest results were computed and discarded
- No historical trade log, no strategy comparison over time
- ML predictions were mocked / not stored

### After (Persistent + Analytical)
- **Cached backtests**: identical parameters return instantly from DB
- **Historical trade log**: query every trade ever executed
- **Strategy comparison**: aggregate metrics across all runs
- **Model accuracy tracking**: store predictions + actuals to compute error over time
- **Time-series price storage**: indexed `(stock_id, date)` for fast range scans

---

## 🛠 Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI (Python), Pandas, NumPy, Scikit-Learn, TensorFlow, XGBoost |
| **Database** | PostgreSQL 15, SQLAlchemy 2.0 ORM, Alembic (migrations) |
| **Frontend** | React.js, Vite, Recharts (Visualization), Axios |
| **Data Source** | Yahoo Finance API (via `yfinance`) |
| **DevOps** | Docker, Docker Compose |

---

## 🗄 Database Schema

```
┌─────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│     stocks      │     │   price_history    │     │  backtest_runs   │
├─────────────────┤     ├────────────────────┤     ├──────────────────┤
│ id (PK)         │◄────┤ stock_id (FK)      │     │ id (PK)          │
│ ticker (UQ)     │     │ date               │     │ stock_id (FK)    │◄──┐
│ name            │     │ open, high, low    │     │ strategy         │   │
│ exchange        │     │ close, volume      │     │ start_date       │   │
│ created_at      │     │ created_at         │     │ end_date         │   │
└─────────────────┘     └────────────────────┘     │ initial_capital  │   │
                                                 │ final_capital    │   │
                                                 │ total_return     │   │
                                                 │ sharpe_ratio     │   │
                                                 │ max_drawdown     │   │
                                                 │ calmar_ratio     │   │
                                                 │ win_rate         │   │
                                                 │ total_trades     │   │
                                                 │ volatility       │   │
                                                 │ created_at       │   │
                                                 └──────────────────┘   │
                                                          │               │
                              ┌───────────────────────────┘               │
                              │                                           │
                              ▼                                           │
                    ┌─────────────────┐     ┌─────────────────────────┐     │
                    │     trades      │     │  portfolio_snapshots    │     │
                    ├─────────────────┤     ├─────────────────────────┤     │
                    │ id (PK)         │     │ id (PK)                 │     │
                    │ backtest_run_id │◄────┤ backtest_run_id (FK)    │     │
                    │ stock_id (FK)   │     │ date                    │     │
                    │ date            │     │ strategy_value          │     │
                    │ type (BUY/SELL) │     │ benchmark_value         │     │
                    │ price           │     │ created_at              │     │
                    │ shares          │     └─────────────────────────┘     │
                    │ order_value     │                                       │
                    │ pnl             │                                       │
                    │ created_at      │                                       │
                    └─────────────────┘                                       │
                                                                              │
                                                 ┌────────────────────────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │  model_predictions  │
                                      ├─────────────────────┤
                                      │ id (PK)             │
                                      │ stock_id (FK)       │
                                      │ model_type          │
                                      │ predicted_value     │
                                      │ predicted_signal    │
                                      │ actual_price        │
                                      │ prediction_date     │
                                      │ horizon_days        │
                                      │ accuracy            │
                                      │ created_at          │
                                      └─────────────────────┘
```

### Indexing Rationale

| Table | Index | Columns | Why |
|-------|-------|---------|-----|
| `price_history` | `ix_price_history_stock_date` | `(stock_id, date)` | **Critical**. Time-series table queried almost exclusively by stock + date range. Composite index enables fast range scans for charting and indicator computation. |
| `price_history` | `ix_price_history_date` | `(date)` | Supports cross-stock analytics (e.g., "what happened on market crash days"). |
| `backtest_runs` | `ix_backtest_runs_stock_strategy` | `(stock_id, strategy)` | Cache-lookup index: we check `(stock, strategy, start, end, capital)` on every backtest request. |
| `backtest_runs` | `ix_backtest_runs_created` | `(created_at)` | Dashboard "recent backtests" query sorts by this. |
| `trades` | `ix_trades_backtest_date` | `(backtest_run_id, date)` | Trade log per backtest; ordered by date for equity-curve reconstruction. |
| `trades` | `ix_trades_stock_date` | `(stock_id, date)` | Cross-backtest trade analysis (e.g., "all AAPL trades across every strategy"). |
| `portfolio_snapshots` | `ix_portfolio_backtest_date` | `(backtest_run_id, date)` | Equity curve endpoint does a range scan per backtest. |
| `model_predictions` | `ix_predictions_stock_date` | `(stock_id, prediction_date)` | Accuracy-over-time queries filter by stock and date range. |
| `model_predictions` | `ix_predictions_model` | `(model_type, prediction_date)` | "How has LSTM accuracy trended globally?" |

---

## 💡 How it Differs from Traditional Trading Apps

| Feature | Retail Trading Apps (e.g., Robinhood) | This Engine |
| :--- | :--- | :--- |
| **Primary Goal** | Executing manual trades | Strategy development & validation |
| **Analysis** | Basic charts & news | Multi-indicator backtesting & AI forecasting |
| **Strategy** | "Gut feeling" or simple alerts | Mathematical proof of edge via historical simulation |
| **Intelligence** | Static indicators | Self-training LSTM & XGBoost ensemble |
| **Focus** | User interface & gamification | Quantitative metrics & risk management |
| **Persistence** | Order history only | Full backtest replay, trade logs, model accuracy tracking |

---

## 📦 Installation & Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ (for local dev)
- Node.js 18+ (for local frontend dev)

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repo
git clone <repo-url>
cd Algorithmic-Trading-Engine

# Start everything (Postgres + Backend + Frontend)
docker-compose up --build

# In a separate terminal, initialize DB tables
docker-compose exec backend python backend/scripts/init_db.py

# Optional: backfill with historical data
docker-compose exec backend python backend/scripts/backfill.py
```

Services will be available at:
- **Frontend**: http://localhost:80
- **Backend API**: http://localhost:8000
- **Postgres**: localhost:5432 (user: `algo`, password: `algo`, db: `trading_engine`)

### Option 2: Local Development

#### 1. Database
```bash
# Start Postgres locally (or use Docker)
docker run -d --name trading-db \
  -e POSTGRES_USER=algo \
  -e POSTGRES_PASSWORD=algo \
  -e POSTGRES_DB=trading_engine \
  -p 5432:5432 postgres:15-alpine
```

#### 2. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create tables
python scripts/init_db.py

# Run server
python api/main.py
```

#### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🔌 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/backtest` | Run a backtest (cached if already exists) |
| `GET` | `/api/compare` | Compare all 3 strategies on a ticker |
| `GET` | `/api/predict/{ticker}` | LSTM + XGBoost ensemble prediction |

### Dashboard / Analytics Endpoints (DB-backed)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/equity_curve/{backtest_id}` | Portfolio value over time for a backtest |
| `GET` | `/api/trade_log/{backtest_id}` | All trades for a backtest run |
| `GET` | `/api/strategy_comparison` | Average metrics per strategy across all runs |
| `GET` | `/api/historical_backtests` | List recent backtest runs |
| `GET` | `/api/top_trades` | Top N most profitable trades |
| `GET` | `/api/prediction_accuracy/{ticker}` | Model accuracy over time |

---

## 🧪 Backfill / Seeding

The backfill script replays backtests across **2022-01-01 → 2023-12-31** for **5 stocks** (AAPL, TSLA, NVDA, MSFT, GOOGL) across **3 strategies** (EMA, RSI, MACD), plus generates LSTM/XGBoost predictions. This populates ~15 backtest runs, ~100+ trades, and ~10 predictions so that dashboard SQL queries return meaningful results immediately.

```bash
cd backend
python scripts/backfill.py
```

---

## 🗺 Roadmap

- [x] Postgres + SQLAlchemy ORM schema
- [x] Docker Compose with db service
- [x] Persist backtest runs, trades, portfolio snapshots
- [x] Persist model predictions with accuracy tracking
- [x] DB-backed dashboard endpoints (equity curve, trade log, strategy comparison)
- [x] Backfill script for realistic demo data
- [ ] Alembic migrations for schema versioning
- [ ] Real-time price streaming via WebSocket
- [ ] Portfolio rebalancing engine
- [ ] Multi-asset correlation matrix

---

## 🛡 Disclaimer

This software is for **educational and research purposes only**. Algorithmic trading involves significant risk. Never trade with money you cannot afford to lose. The AI predictions are mathematical estimations and do not guarantee future market performance.

---

Built with 💚 by [Naman Rai](https://github.com/namanraii)
