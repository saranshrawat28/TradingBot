"""
Comprehensive Verification and Test Suite for Indian Algorithmic Trading Bot.
"""

import sys
import os
import io

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.strategies.indicators import (
    calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_supertrend, calculate_atr, add_all_indicators
)
from src.data.data_fetcher import get_historical_data, get_live_quote
from src.strategies import AVAILABLE_STRATEGIES, get_strategy
from src.brokers.paper_broker import PaperBroker
from src.engine.risk_manager import RiskManager
from src.engine.backtester import Backtester
from src.utils.helpers import is_market_open, format_currency_inr, clean_symbol

def test_indicators():
    print("[1/6] Testing Technical Indicators...")
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100)))
    high = prices + 1.5
    low = prices - 1.5
    close = prices
    volume = pd.Series(10000, index=prices.index)
    
    ema9 = calculate_ema(close, 9)
    rsi14 = calculate_rsi(close, 14)
    macd, sig, hist = calculate_macd(close, 12, 26, 9)
    up, mid, low_b, width, pct_b = calculate_bollinger_bands(close, 20, 2.0)
    st, st_dir = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
    
    assert len(ema9) == 100, "EMA length mismatch"
    assert len(rsi14) == 100, "RSI length mismatch"
    assert len(macd) == 100, "MACD length mismatch"
    assert len(up) == 100, "BB length mismatch"
    assert len(st) == 100, "SuperTrend length mismatch"
    assert len(atr) == 100, "ATR length mismatch"
    print(" -> All indicators calculated successfully!")

def test_data_fetcher():
    print("[2/6] Testing Data Fetcher...")
    df = get_historical_data("RELIANCE.NS", period="1mo", interval="1d")
    assert not df.empty, "Historical data returned empty"
    assert "Close" in df.columns, "Missing Close column"
    
    quote = get_live_quote("RELIANCE.NS")
    assert "price" in quote, "Missing price in live quote"
    print(f" -> Fetched RELIANCE.NS data: {len(df)} bars, Live Price: ₹{quote['price']:.2f}")

def test_strategies():
    print("[3/6] Testing Strategy Signal Generation...")
    df = get_historical_data("TCS.NS", period="3mo", interval="1d")
    
    for name, strat_cls in AVAILABLE_STRATEGIES.items():
        strat = strat_cls()
        res_df = strat.generate_signals(df)
        assert "Signal" in res_df.columns, f"Missing Signal column in {name}"
        signals_count = (res_df["Signal"] != 0).sum()
        print(f" -> Strategy '{name}': Generated {signals_count} signals")

def test_paper_broker():
    print("[4/6] Testing Paper Broker Execution & Indian Tax Modeling...")
    broker = PaperBroker(initial_capital=100000.0)
    bal_before = broker.get_account_balance()
    
    # Place Buy Order
    order = broker.place_order(
        symbol="SBIN.NS",
        side="BUY",
        quantity=10,
        price=800.0,
        sl=780.0,
        tp=840.0,
        strategy_name="Unit Test"
    )
    assert order["status"] == "FILLED", "Buy order failed"
    assert "tax_breakdown" in order, "Missing tax breakdown"
    print(f" -> Buy Order Executed: {order['quantity']} qty @ ₹{order['price']:.2f}, Taxes: ₹{order['fee']:.2f}")
    
    positions = broker.get_open_positions()
    assert len(positions) >= 1, "Position not recorded in database"
    
    # Square off
    sq_res = broker.square_off_position("SBIN.NS", reason="Test Square-off")
    assert sq_res["status"] == "SUCCESS", "Square-off failed"
    print(f" -> Square-Off Executed: Net PnL: ₹{sq_res['net_pnl']:.2f}")

def test_risk_manager():
    print("[5/6] Testing Risk Manager...")
    rm = RiskManager(risk_per_trade_pct=2.0, default_sl_pct=1.5, default_tp_pct=3.0)
    qty = rm.calculate_position_size(total_equity=100000.0, entry_price=2500.0, stop_loss_price=2462.5)
    sl, tp = rm.calculate_sl_tp_prices("BUY", entry_price=2500.0)
    
    assert qty > 0, "Position size should be > 0"
    assert sl < 2500.0, "SL should be below entry for BUY"
    assert tp > 2500.0, "TP should be above entry for BUY"
    print(f" -> Position Sizing: {qty} shares | SL: ₹{sl:.2f} | TP: ₹{tp:.2f}")

def test_backtester():
    print("[6/6] Testing Backtester Simulation Engine...")
    df = get_historical_data("INFY.NS", period="6mo", interval="1d")
    strat = get_strategy("EMA Crossover + RSI")
    backtester = Backtester(strategy=strat, initial_capital=100000.0)
    results = backtester.run(df)
    
    assert "total_return_pct" in results, "Missing total return"
    assert "win_rate_pct" in results, "Missing win rate"
    assert "trades" in results, "Missing trade list"
    assert "equity_df" in results, "Missing equity series"
    print(f" -> Backtest Finished: Return: {results['total_return_pct']:+.2f}%, Win Rate: {results['win_rate_pct']:.1f}%, Trades: {results['total_trades']}")

if __name__ == "__main__":
    print("==================================================")
    print("  RUNNING APEXTRADE FULL SYSTEM VERIFICATION")
    print("==================================================")
    test_indicators()
    test_data_fetcher()
    test_strategies()
    test_paper_broker()
    test_risk_manager()
    test_backtester()
    print("==================================================")
    print("  ALL 6 TEST MODULES PASSED PERFECTLY! 🚀")
    print("==================================================")
