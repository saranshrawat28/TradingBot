# ApexTrade — Indian Stocks & F&O Algorithmic Trading Bot & Web Terminal 📈🇮🇳

An institutional-grade, modular Algorithmic Trading Bot and Web Dashboard built specifically for **Indian Equities & F&O (NSE / BSE)**. Features zero-risk realistic paper trading, multi-indicator strategies, dynamic risk management, backtesting analytics, and seamless broker connectivity (Zerodha Kite Connect, Angel One SmartAPI, DhanHQ).

---

## 🌟 Key Features

* **🛡️ Zero-Risk Paper Trading Broker (Default)**:
  - Realistic order matching with Indian regulatory taxes and charges modeled:
    - Brokerage (₹20 or 0.03% intraday)
    - STT (Securities Transaction Tax)
    - Exchange Turnover Charges (NSE)
    - SEBI Turnover Charges
    - Stamp Duty
    - GST (18%)
    - Execution Slippage simulation (0.05%)
* **📊 Quantitative Strategy Suite**:
  - **EMA Crossover + RSI Momentum**: Fast/Slow EMA trend cross confirmed with RSI momentum and 200 EMA macro-trend filter.
  - **MACD Momentum**: Moving average convergence divergence histogram expansion and zero-line crossovers.
  - **Bollinger Bands Dynamic Strategy**: Mean reversion bounce and volatility breakout modes.
  - **SuperTrend Intraday/Swing**: ATR-based dynamic trailing trend-following system.
  - **Multi-Indicator Confluence**: Institutional multi-factor setup combining Trend (EMA) + Momentum (RSI/MACD) + Volatility (SuperTrend).
* **🛡️ Institutional Risk Management**:
  - Capital allocation per trade & ATR volatility position sizing.
  - Dynamic Stop-Loss (SL) and Take-Profit (TP) levels.
  - Real-time Trailing Stop-Loss adjustments.
  - Daily Drawdown Circuit Breaker (halts execution on max loss).
  - Automated **3:15 PM IST Intraday Square-Off** safeguard.
* **📈 Interactive Backtesting & Optimizer**:
  - Full candlestick chart with Buy/Sell execution marker flags (Plotly).
  - Performance KPI metrics: Total Return %, Benchmark Return % (NIFTY 50), Win Rate %, Profit Factor, Sharpe Ratio, Sortino Ratio, Max Drawdown %.
  - Complete trade history journal with CSV export.
* **🔍 NIFTY 50 Live Screener & Signal Radar**:
  - Scans top NSE equities in real-time and calculates algorithmic bias (Strong Buy, Buy, Neutral, Sell, Strong Sell).
* **🔌 Multi-Broker Adapters**:
  - Paper Broker (Built-in)
  - Zerodha Kite Connect (`kiteconnect`)
  - Angel One SmartAPI (`smartapi-python`)
  - DhanHQ (`dhanhq`)

---

## 🚀 Quick Start

### 1. Run via Batch Script (Windows)
Double-click `run.bat` or run:
```powershell
python -m streamlit run app.py
```

### 2. Open Terminal
Navigate to [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Architecture

```
TradingBot/
├── app.py                     # Streamlit Dark Terminal Dashboard
├── config.py                  # Watchlist, fees, risk rules, and broker settings
├── requirements.txt           # Dependencies
├── run.bat                    # Windows launcher
├── .env.example               # Broker API keys template
│
└── src/
    ├── brokers/               # Broker Adapter Layer
    │   ├── base_broker.py     # Base abstract interface
    │   ├── paper_broker.py    # Simulated paper broker with Indian fees
    │   ├── zerodha_broker.py  # Kite Connect adapter
    │   ├── angel_broker.py    # Angel One SmartAPI adapter
    │   └── dhan_broker.py     # DhanHQ adapter
    │
    ├── strategies/            # Strategy & Indicator Suite
    │   ├── indicators.py      # Vectorized pure NumPy/Pandas indicators
    │   ├── base_strategy.py   # Strategy interface
    │   ├── ema_rsi_strategy.py# EMA + RSI momentum strategy
    │   ├── macd_strategy.py   # MACD strategy
    │   ├── bollinger_strategy.py # Bollinger Bands strategy
    │   ├── supertrend_strategy.py# SuperTrend strategy
    │   └── multi_indicator.py # Multi-confluence strategy
    │
    ├── engine/                # Execution & Risk Layer
    │   ├── risk_manager.py    # Position sizing, SL/TP, Trailing SL, Circuit breaker
    │   ├── backtester.py      # Backtesting and performance analytics
    │   └── live_bot.py        # Automated live scanning and execution loop
    │
    ├── data/                  # Market Data Fetcher
    │   └── data_fetcher.py    # Multi-timeframe historical & real-time quotes
    │
    └── utils/                 # Storage & Helpers
        ├── helpers.py         # IST clock, market hours, INR currency formatting
        └── storage.py         # SQLite persistence for orders, positions, and trades
```

---

## 🔒 Broker API Configuration (Optional for Live Trading)

To connect your live broker account:
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Enter your broker credentials in `.env` or in the **⚙️ Broker & Risk Settings** tab of the dashboard.
3. Switch active broker from `paper` to `zerodha`, `angel`, or `dhan`.

---

## ⚖️ Disclaimer
*This software is intended for educational and algorithmic research purposes. Algorithmic and manual trading in equities, derivatives, and financial markets involves substantial risk of loss. Always test your strategies thoroughly in Paper Trading Mode before deploying live capital.*
