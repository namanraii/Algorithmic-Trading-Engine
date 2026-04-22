import React, { useState } from 'react';
import axios from 'axios';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

// Separate axios instance with timeout
const api = axios.create({ baseURL: API_BASE, timeout: 60000 });

const QUICK_PICKS = [
  { symbol: 'AAPL',          name: 'Apple Inc.',               price: '$189', change: '+1.24%', positive: true },
  { symbol: 'TSLA',          name: 'Tesla Inc.',               price: '$142', change: '-0.87%', positive: false },
  { symbol: 'NVDA',          name: 'NVIDIA Corp.',             price: '$875', change: '+3.21%', positive: true },
  { symbol: 'RELIANCE.NS',   name: 'Reliance Industries',      price: '₹2934',change: '+0.55%', positive: true },
  { symbol: 'INFY.NS',       name: 'Infosys Ltd (NSE)',        price: '₹1452',change: '-0.82%', positive: false },
  { symbol: 'BTC-USD',       name: 'Bitcoin / USD',            price: '$64k', change: '-0.33%', positive: false },
  { symbol: 'HINDCOPPER.NS', name: 'Hindustan Copper',         price: '₹556', change: '+0.92%', positive: true },
];

const STRATEGIES = [
  { value: 'ema',  label: 'EMA Crossover',      desc: 'Buy when fast EMA crosses above slow EMA. Classic trend-following.' },
  { value: 'rsi',  label: 'RSI Mean Reversion', desc: 'Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought).' },
  { value: 'macd', label: 'MACD Momentum',      desc: 'Capture momentum when MACD line crosses the signal line.' },
];

export default function App() {
  const [ticker,     setTicker]     = useState('AAPL');
  const [strategy,   setStrategy]   = useState('ema');
  const [startDate,  setStartDate]  = useState('2022-01-01');
  const [endDate,    setEndDate]    = useState('2023-12-31');
  const [capital,    setCapital]    = useState(100000);
  const [loading,    setLoading]    = useState(false);
  const [mlLoading,  setMlLoading]  = useState(false);
  const [results,    setResults]    = useState(null);
  const [mlPred,     setMlPred]     = useState(null);
  const [error,      setError]      = useState(null);
  const [activePick, setActivePick] = useState(null);
  const [searchVal,  setSearchVal]  = useState('');

  const runBacktest = async (overrideTicker) => {
    const t = (overrideTicker || ticker).trim().toUpperCase();
    if (!t) return;

    setTicker(t);
    setActivePick(t);
    setLoading(true);
    setResults(null);
    setMlPred(null);
    setError(null);

    try {
      const res = await api.post('backtest', {
        ticker: t,
        strategy,
        start_date: startDate,
        end_date: endDate,
        initial_capital: capital,
      });
      setResults(res.data);

      // Run ML prediction SEPARATELY so it doesn't block the chart from rendering
      fetchMlPrediction(t);
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message || 'Unknown error';
      setError(`Backtest failed: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchMlPrediction = async (t) => {
    setMlLoading(true);
    try {
      const ml = await api.get(`predict/${t}`);
      if (ml.data && !ml.data.error) setMlPred(ml.data);
    } catch (_) {
      // ML is optional — silently ignore
    } finally {
      setMlLoading(false);
    }
  };

  const handleSearch = (e) => {
    if (e.key === 'Enter' && searchVal.trim()) {
      runBacktest(searchVal.trim().toUpperCase());
      setSearchVal('');
    }
  };

  const handleQuickPick = (symbol) => runBacktest(symbol);

  const ret = results?.metrics?.total_return;
  const stratLabel = STRATEGIES.find(s => s.value === strategy)?.label;

  return (
    <>
      {/* ── SIDEBAR ── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1>Algorithmic<br />Trading Engine</h1>
          <span>FastAPI · LSTM · XGBoost</span>
        </div>

        <div className="sidebar-search">
          <div className="sidebar-search-inner">
            <span>🔍</span>
            <input
              type="text"
              placeholder="Type ticker, press Enter…"
              value={searchVal}
              onChange={e => setSearchVal(e.target.value.toUpperCase())}
              onKeyDown={handleSearch}
            />
          </div>
          <div style={{ fontSize: '0.63rem', color: 'var(--text-muted)', marginTop: '5px', paddingLeft: '2px' }}>
            AAPL · TSLA · BTC-USD · INFY.NS · ^NSEI
          </div>
        </div>

        <div className="sidebar-section-title">Popular Tickers</div>

        <div className="sidebar-stocks">
          {QUICK_PICKS.map(p => (
            <div
              key={p.symbol}
              className={`stock-row ${activePick === p.symbol ? 'active' : ''}`}
              onClick={() => handleQuickPick(p.symbol)}
            >
              <div className="stock-row-left">
                <span className="stock-row-symbol">{p.symbol}</span>
                <span className="stock-row-name">{p.name}</span>
              </div>
              <div className="stock-row-right">
                <div className="stock-row-price">{p.price}</div>
                <span className={`stock-row-badge ${p.positive ? 'badge-green' : 'badge-red'}`}>
                  {p.change}
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="sidebar-howto">
          <h3>📖 How To Use</h3>
          <div className="howto-step">
            <div className="howto-num">1</div>
            <div className="howto-text"><strong>Pick a ticker</strong> — click a stock or type a symbol &amp; press Enter.</div>
          </div>
          <div className="howto-step">
            <div className="howto-num">2</div>
            <div className="howto-text"><strong>Choose strategy</strong> — EMA (trend), RSI (reversion), MACD (momentum).</div>
          </div>
          <div className="howto-step">
            <div className="howto-num">3</div>
            <div className="howto-text"><strong>Run Backtest</strong> — review portfolio chart, metrics &amp; AI signals.</div>
          </div>
        </div>
      </aside>

      {/* ── MAIN ── */}
      <main className="main">

        {/* Top bar */}
        <div className="main-topbar">
          <div>
            <div className="ticker-title">{ticker}</div>
            <div className="ticker-sub">
              {loading  ? 'Fetching data from Yahoo Finance…' :
               results  ? `${stratLabel} · ${startDate} → ${endDate} · $${capital.toLocaleString()} capital` :
               'Select a ticker or type one in the search bar to begin'}
            </div>
          </div>
          {mlPred && (
            <div className="price-area">
              <div className="price-big" style={{ color: mlPred.predicted_price_lstm > mlPred.current_price ? 'var(--green)' : 'var(--red)' }}>
                ${mlPred.current_price.toFixed(2)}
              </div>
              <div className="price-change" style={{ color: mlPred.predicted_price_lstm > mlPred.current_price ? 'var(--green)' : 'var(--red)' }}>
                AI Target: ${mlPred.predicted_price_lstm.toFixed(2)}
              </div>
            </div>
          )}
          {mlLoading && !mlPred && (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>🤖 AI computing…</div>
          )}
        </div>

        {/* Controls */}
        <div className="controls-bar">
          <div className="ctrl-group">
            <div className="ctrl-label">Ticker Symbol</div>
            <input
              className="ctrl-input"
              type="text"
              value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === 'Enter' && runBacktest()}
              placeholder="e.g. AAPL"
              style={{ width: '120px' }}
            />
          </div>
          <div className="ctrl-group">
            <div className="ctrl-label">Strategy</div>
            <select className="ctrl-select" value={strategy} onChange={e => setStrategy(e.target.value)}>
              {STRATEGIES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </div>
          <div className="ctrl-group">
            <div className="ctrl-label">Start Date</div>
            <input className="ctrl-input" type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
          </div>
          <div className="ctrl-group">
            <div className="ctrl-label">End Date</div>
            <input className="ctrl-input" type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
          </div>
          <div className="ctrl-group">
            <div className="ctrl-label">Capital ($)</div>
            <input className="ctrl-input" type="number" value={capital} onChange={e => setCapital(Number(e.target.value))} style={{ width: '110px' }} />
          </div>
          <div style={{ flexGrow: 1 }} />
          <button className="run-btn" onClick={() => runBacktest()} disabled={loading}>
            {loading ? '⏳ Running…' : '▶ Run Backtest'}
          </button>
        </div>

        {/* Content */}
        <div className="content">

          {/* Error */}
          {error && (
            <div style={{ background: 'var(--red-dim)', border: '1px solid var(--red)', borderRadius: '10px', padding: '16px 20px', color: 'var(--red)', fontSize: '0.85rem' }}>
              ⚠ {error}
            </div>
          )}

          {/* AI Forecast */}
          {mlPred && (
            <div className="forecast-bar">
              <div>
                <div className="forecast-tag">XGBoost Signal</div>
                <div className="forecast-val" style={{ color: mlPred.signal_xgboost === 'BUY' ? 'var(--green)' : 'var(--red)' }}>
                  {mlPred.signal_xgboost === 'BUY' ? '⚡ BUY' : '⚠ HOLD / SELL'}
                </div>
              </div>
              <div className="forecast-divider" />
              <div>
                <div className="forecast-tag">LSTM Predicted Price</div>
                <div className="forecast-val" style={{ color: mlPred.predicted_price_lstm > mlPred.current_price ? 'var(--green)' : 'var(--red)' }}>
                  ${mlPred.predicted_price_lstm.toFixed(2)}
                </div>
              </div>
              <div className="forecast-divider" />
              <div>
                <div className="forecast-tag">Current Price</div>
                <div className="forecast-val">${mlPred.current_price.toFixed(2)}</div>
              </div>
              <div className="forecast-note">🤖 LSTM-RNN + XGBoost Ensemble · Yahoo Finance data</div>
            </div>
          )}

          {/* Chart */}
          {results && (
            <div className="chart-card">
              <div className="chart-header">
                <div className="chart-title">Portfolio Performance vs Buy &amp; Hold Benchmark</div>
                <div className="chart-legend">
                  <span>● <span style={{ color: 'var(--green)' }}>Your Strategy</span></span>
                  <span>● <span style={{ color: '#444' }}>Buy &amp; Hold</span></span>
                </div>
              </div>
              <ResponsiveContainer width="100%" height="88%">
                <AreaChart data={results.portfolio} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="gStrategy" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#00c805" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#00c805" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1c1c1c" vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: '#555', fontSize: 10 }} tickFormatter={t => t.split(' ')[0]} />
                  <YAxis orientation="right" tick={{ fill: '#555', fontSize: 10 }} domain={['auto', 'auto']} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '8px', fontSize: '12px' }}
                    formatter={v => [`$${parseFloat(v).toFixed(2)}`]}
                    labelFormatter={l => l.split(' ')[0]}
                  />
                  <Area type="monotone" dataKey="value"     stroke="#00c805" strokeWidth={2} fill="url(#gStrategy)" name="Strategy" />
                  <Area type="monotone" dataKey="benchmark" stroke="#444"    strokeDasharray="4 4" fill="none" name="Buy & Hold" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Metrics */}
          {results && (
            <div className="metrics-grid">
              <MetricCell label="Total Return"  value={`${ret >= 0 ? '+' : ''}${ret}%`}                        color={ret >= 0 ? 'var(--green)' : 'var(--red)'}    sub={`Starting capital: $${capital.toLocaleString()}`} />
              <MetricCell label="Sharpe Ratio"  value={results.metrics.sharpe_ratio}                           color={results.metrics.sharpe_ratio > 1 ? 'var(--green)' : '#fff'} sub="> 1.0 is strong risk-adjusted return" />
              <MetricCell label="Win Rate"      value={`${results.metrics.win_rate}%`}                         color={results.metrics.win_rate > 50 ? 'var(--green)' : 'var(--red)'} sub={`${results.metrics.total_trades} total trades executed`} />
              <MetricCell label="Max Drawdown"  value={`${results.metrics.max_drawdown}%`}                     color="var(--red)"                                  sub="Worst peak-to-trough loss" />
            </div>
          )}

          {/* Additional metrics row */}
          {results && (
            <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(3,1fr)' }}>
              <MetricCell label="Volatility"   value={`${results.metrics.volatility}%`}   color="#fff" sub="Annualised standard deviation" />
              <MetricCell label="Calmar Ratio" value={results.metrics.calmar_ratio}        color="#fff" sub="Annual return ÷ max drawdown" />
              <MetricCell label="Total Trades" value={results.metrics.total_trades}        color="#fff" sub="Buy executions during period" />
            </div>
          )}

          {/* Trade Log */}
          {results && results.trades.length > 0 && (
            <div className="trade-card">
              <div className="trade-card-header">Recent Trade Activity — {results.trades.length} total executions</div>
              <table className="trade-table">
                <thead>
                  <tr>
                    <th>Date</th><th>Action</th><th>Exec Price</th><th>Shares</th><th>Order Value</th>
                  </tr>
                </thead>
                <tbody>
                  {results.trades.slice(-8).reverse().map((t, i) => (
                    <tr key={i}>
                      <td style={{ color: 'var(--text-secondary)' }}>{t.date.split(' ')[0]}</td>
                      <td><span className={t.type === 'BUY' ? 'tag-buy' : 'tag-sell'}>{t.type}</span></td>
                      <td style={{ fontWeight: 600 }}>${t.price.toFixed(2)}</td>
                      <td>{t.shares}</td>
                      <td>${(t.price * t.shares).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {results.trades.length > 8 && (
                <div style={{ padding: '12px 20px', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                  Showing last 8 of {results.trades.length} trades
                </div>
              )}
            </div>
          )}

          {/* Welcome */}
          {!results && !loading && !error && (
            <div className="welcome">
              <div className="welcome-icon">📈</div>
              <h2>Welcome to the Algorithmic Trading Engine</h2>
              <p>
                Fetch real market data from <strong>Yahoo Finance</strong>, simulate your chosen trading strategy
                across historical data, compute professional quant metrics (Sharpe Ratio, Max Drawdown, Win Rate),
                and get AI-powered price predictions from our <strong>LSTM + XGBoost</strong> ensemble.
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                Click any ticker in the sidebar, type a symbol in the search bar, or click a chip below.
              </p>

              <div className="ticker-examples">
                {['AAPL', 'TSLA', 'BTC-USD', 'INFY.NS', 'NVDA', 'RELIANCE.NS'].map(sym => (
                  <div key={sym} className="ticker-chip" onClick={() => handleQuickPick(sym)}>
                    {sym}
                    <span className="chip-cat">
                      {sym.endsWith('.NS') ? 'NSE India' : sym.includes('-') ? 'Crypto' : 'US Stock'}
                    </span>
                  </div>
                ))}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '16px', width: '100%', maxWidth: '720px', marginTop: '20px' }}>
                {STRATEGIES.map(s => (
                  <div key={s.value} style={{ background: 'var(--bg-1)', border: '1px solid var(--border)', borderRadius: '10px', padding: '16px' }}>
                    <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--green)', marginBottom: '6px' }}>{s.label}</div>
                    <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{s.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="welcome">
              <div style={{ fontSize: '2.5rem' }}>⏳</div>
              <h2>Running {stratLabel} on {ticker}…</h2>
              <p style={{ color: 'var(--text-muted)' }}>
                Downloading OHLCV data from Yahoo Finance, computing {15}+ technical indicators,
                simulating trades with 0.1% commission, and calculating performance metrics.
              </p>
            </div>
          )}

        </div>
      </main>
    </>
  );
}

function MetricCell({ label, value, color, sub }) {
  return (
    <div className="metric-cell">
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={{ color }}>{value}</div>
      <div className="metric-sub">{sub}</div>
    </div>
  );
}
