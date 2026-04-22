# Algorithmic Trading Engine with AI Forecasting 📈

A high-performance, full-stack platform designed for quantitative traders and developers to build, test, and analyze algorithmic trading strategies. Unlike traditional retail trading apps that focus solely on order execution, this engine provides a comprehensive environment for **Strategy Validation** and **AI-Driven Prediction**.

## 🔗 Live Demo
- **Dashboard**: [https://algorithmic-trading-engine.vercel.app/](https://algorithmic-trading-engine.vercel.app/)
- **API Backend**: [https://algorithmic-trading-engine.onrender.com/](https://algorithmic-trading-engine.onrender.com/)

![Dashboard Preview](https://raw.githubusercontent.com/namanraii/Algorithmic-Trading-Engine/main/docs/dashboard_preview.png) *(Placeholder for preview)*

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

## 🛠 Technology Stack

- **Backend**: FastAPI (Python), Pandas, NumPy, Scikit-Learn, TensorFlow, XGBoost.
- **Frontend**: React.js, Vite, Recharts (Visualization), Axios.
- **Data Source**: Yahoo Finance API (via `yfinance`).

## 💡 How it Differs from Traditional Trading Apps

| Feature | Retail Trading Apps (e.g., Robinhood) | This Engine |
| :--- | :--- | :--- |
| **Primary Goal** | Executing manual trades | Strategy development & validation |
| **Analysis** | Basic charts & news | Multi-indicator backtesting & AI forecasting |
| **Strategy** | "Gut feeling" or simple alerts | Mathematical proof of edge via historical simulation |
| **Intelligence** | Static indicators | Self-training LSTM & XGBoost ensemble |
| **Focus** | User interface & gamification | Quantitative metrics & risk management |

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python api/main.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🛡 Disclaimer
This software is for **educational and research purposes only**. Algorithmic trading involves significant risk. Never trade with money you cannot afford to lose. The AI predictions are mathematical estimations and do not guarantee future market performance.

---
Built with 💚 by [Naman Rai](https://github.com/namanraii)
