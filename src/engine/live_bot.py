"""
Real-Time Automated Trading Bot Engine.
Coordinates data scanning, strategy signal evaluation, risk checks, and broker order execution.
"""

from datetime import datetime
import time
import threading
import config
from src.brokers import get_broker
from src.strategies import get_strategy
from src.engine.risk_manager import RiskManager
from src.data.data_fetcher import get_historical_data, get_live_quote
from src.utils.helpers import get_ist_now, is_market_open, is_intraday_squareoff_time

class LiveTradingBot:
    """
    Automated trading controller.
    """
    
    def __init__(
        self,
        strategy_name: str = "EMA Crossover + RSI",
        strategy_params: dict = None,
        symbols: list = None,
        broker_name: str = "paper",
        timeframe: str = "5m",
        initial_capital: float = config.DEFAULT_INITIAL_CAPITAL
    ):
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        self.strategy = get_strategy(strategy_name, **self.strategy_params)
        self.broker_name = broker_name
        self.broker = get_broker(broker_name)
        self.symbols = symbols or [item["symbol"] for item in config.DEFAULT_WATCHLIST[:10]]
        self.timeframe = timeframe
        self.risk_manager = RiskManager()
        
        self.is_running = False
        self._thread: threading.Thread = None
        self.last_scan_time = None
        self.logs = []
        self.recent_signals = []

    def start_continuous(self, interval_sec: int = 60):
        """Start continuous background trading thread."""
        if self.is_running:
            return
        self.is_running = True
        self.log(f"🟢 Background Bot Engine started. Scanning every {interval_sec}s.", level="SUCCESS")
        self._thread = threading.Thread(target=self._continuous_loop, args=(interval_sec,), daemon=True)
        self._thread.start()

    def stop_continuous(self):
        """Stop background trading thread."""
        self.is_running = False
        self.log("⏸️ Background Bot Engine stopped.", level="WARNING")

    def _continuous_loop(self, interval_sec: int):
        while self.is_running:
            try:
                self.scan_and_execute()
            except Exception as e:
                self.log(f"Continuous scan error: {e}", level="ERROR")
            
            for _ in range(max(1, int(interval_sec))):
                if not self.is_running:
                    break
                time.sleep(1)

    def log(self, message: str, level: str = "INFO"):
        ist_str = get_ist_now().strftime("%H:%M:%S")
        entry = f"[{ist_str}] [{level}] {message}"
        self.logs.insert(0, entry)
        if len(self.logs) > 200:
            self.logs = self.logs[:200]
        try:
            print(entry)
        except Exception:
            try:
                print(entry.encode("ascii", "replace").decode("ascii"))
            except Exception:
                pass

    def scan_and_execute(self) -> dict:
        """
        Execute one complete scan iteration across all watchlist symbols:
        1. Check market hours and auto-squareoff timer.
        2. Update open positions (Trailing SL / SL / TP hits).
        3. Check Circuit Breaker.
        4. Evaluate new signals and place orders.
        """
        self.last_scan_time = get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST")
        market_open, market_msg = is_market_open()
        
        balance_info = self.broker.get_account_balance()
        current_equity = balance_info.get("total_equity", config.DEFAULT_INITIAL_CAPITAL)
        init_cap = balance_info.get("initial_capital", config.DEFAULT_INITIAL_CAPITAL)
        
        # 1. Check Circuit Breaker
        cb_triggered, cb_msg = self.risk_manager.is_daily_circuit_breaker_triggered(init_cap, current_equity)
        if cb_triggered:
            self.log(cb_msg, level="WARNING")
            self.broker.square_off_all(reason="Circuit Breaker")
            return {"status": "HALTED", "message": cb_msg}
            
        # 2. Check 3:15 PM IST Intraday Auto Square-Off
        if is_intraday_squareoff_time():
            self.log("3:15 PM IST reached. Executing automated intraday square-off for all open positions.", level="WARNING")
            sq_results = self.broker.square_off_all(reason="Intraday 15:15 Auto Square-Off")
            return {"status": "SQUARED_OFF", "results": sq_results}
            
        # 3. Update Existing Positions (Check Trailing SL, TP)
        open_positions = self.broker.get_open_positions()
        for pos in open_positions:
            sym = pos["symbol"]
            quote = get_live_quote(sym)
            curr_price = float(quote.get("price", pos["entry_price"]))
            
            # Update Trailing SL
            updated_sl = self.risk_manager.update_trailing_stop(
                pos["side"], pos["entry_price"], curr_price, pos["highest_price"], pos["sl"]
            )
            pos["sl"] = updated_sl
            
            # Check exit conditions
            if pos["side"] == "LONG":
                if pos["sl"] and curr_price <= pos["sl"]:
                    self.log(f"Stop-Loss hit for {sym} at ₹{curr_price:.2f}. Squaring off.", level="INFO")
                    self.broker.square_off_position(sym, reason="Stop-Loss Hit")
                elif pos["tp"] and curr_price >= pos["tp"]:
                    self.log(f"Take-Profit target hit for {sym} at ₹{curr_price:.2f}. Squaring off.", level="SUCCESS")
                    self.broker.square_off_position(sym, reason="Take-Profit Hit")
            elif pos["side"] == "SHORT":
                if pos["sl"] and curr_price >= pos["sl"]:
                    self.log(f"Short Stop-Loss hit for {sym} at ₹{curr_price:.2f}. Squaring off.", level="INFO")
                    self.broker.square_off_position(sym, reason="Stop-Loss Hit")
                elif pos["tp"] and curr_price <= pos["tp"]:
                    self.log(f"Short Take-Profit hit for {sym} at ₹{curr_price:.2f}. Squaring off.", level="SUCCESS")
                    self.broker.square_off_position(sym, reason="Take-Profit Hit")
                    
        # 4. Scan Symbols for New Entry Signals
        signals_found = []
        open_symbols = [p["symbol"] for p in self.broker.get_open_positions()]
        
        for sym in self.symbols:
            if sym.startswith("^"):
                continue  # Skip raw index tickers for direct stock execution
                
            try:
                # Fetch recent candles
                hist = get_historical_data(sym, period="5d", interval=self.timeframe)
                if hist.empty or len(hist) < 20:
                    continue
                    
                latest = self.strategy.get_latest_signal(hist)
                sig = latest["signal"]
                action = latest["action"]
                reason = latest["reason"]
                price = latest["price"]
                
                signal_entry = {
                    "symbol": sym,
                    "action": action,
                    "price": price,
                    "reason": reason,
                    "timestamp": get_ist_now().strftime("%H:%M:%S")
                }
                
                if sig != 0:
                    signals_found.append(signal_entry)
                    self.log(f"Signal Detected on {sym}: {action} @ ₹{price:.2f} ({reason})", level="INFO")
                    
                    # Execute Entry if symbol not already open and max positions not reached
                    if action == "BUY" and sym not in open_symbols:
                        if len(open_symbols) < config.MAX_CONCURRENT_POSITIONS:
                            sl, tp = self.risk_manager.calculate_sl_tp_prices("BUY", price)
                            qty = self.risk_manager.calculate_position_size(balance_info["cash"], price, sl)
                            if qty > 0:
                                order_res = self.broker.place_order(
                                    symbol=sym, side="BUY", quantity=qty, price=price,
                                    sl=sl, tp=tp, strategy_name=self.strategy.name
                                )
                                self.log(f"Order Placed for {sym}: BUY {qty} qty @ ₹{price:.2f} (SL: ₹{sl:.2f}, TP: ₹{tp:.2f})", level="SUCCESS")
                                open_symbols.append(sym)
                                
                    elif action == "SELL" and sym in open_symbols:
                        self.broker.square_off_position(sym, reason=f"Signal: {reason}")
                        self.log(f"Position Closed for {sym} on SELL signal", level="INFO")
                        
            except Exception as e:
                self.log(f"Error scanning {sym}: {e}", level="ERROR")
                
        self.recent_signals = signals_found
        return {
            "status": "SUCCESS",
            "last_scan": self.last_scan_time,
            "signals_count": len(signals_found),
            "signals": signals_found,
            "open_positions": len(self.broker.get_open_positions()),
            "market_open": market_open
        }
