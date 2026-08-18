"""
Autonomous Market Hunter Daemon for Indian Equities.
Continuous 30-second background scanning engine with 2-Stage Multi-Agent Council gating
and Software-Managed OCO (One-Cancels-Other) execution.
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import config
from src.data.data_fetcher import get_historical_data, get_live_quote
from src.ai.multi_agent_council import MultiAgentCouncil
from src.engine.ai_guardrails import AIGuardrails
from src.engine.software_oco_manager import SoftwareOCOManager
from src.utils.storage import get_portfolio_state
from src.utils.helpers import get_ist_now, display_symbol_name

class MarketHunterDaemon:
    """
    Autonomous Background Hunting Daemon for High-Probability Breakout Setups.
    """

    _lock = threading.Lock()
    _thread: Optional[threading.Thread] = None
    _is_running: bool = False
    _logs: List[Dict[str, Any]] = []
    _max_logs: int = 100
    _broker: Any = None
    _scan_interval_sec: int = 30
    _last_scan_time: Optional[str] = None
    _scans_completed: int = 0
    _trades_placed_today: int = 0

    @classmethod
    def start(cls, broker: Any, scan_interval_sec: int = 30):
        """
        Starts the background market hunter daemon.
        """
        with cls._lock:
            if cls._is_running:
                return
            cls._broker = broker
            cls._scan_interval_sec = scan_interval_sec
            cls._is_running = True
            
            # Execute crash recovery on startup
            SoftwareOCOManager.check_and_recover_unhedged_positions(broker)
            
            cls._log_event("DAEMON_START", "Autonomous Market Hunter Daemon initialized and active.")
            cls._thread = threading.Thread(target=cls._run_loop, daemon=True)
            cls._thread.start()

    @classmethod
    def stop(cls):
        """
        Gracefully stops the background hunter daemon.
        """
        with cls._lock:
            cls._is_running = False
            cls._log_event("DAEMON_STOP", "Autonomous Market Hunter Daemon stopped by user.")

    @classmethod
    def is_running(cls) -> bool:
        return cls._is_running

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        return {
            "is_running": cls._is_running,
            "scan_interval_sec": cls._scan_interval_sec,
            "last_scan_time": cls._last_scan_time,
            "scans_completed": cls._scans_completed,
            "trades_placed_today": cls._trades_placed_today,
            "logs": list(cls._logs)
        }

    @classmethod
    def clear_logs(cls):
        with cls._lock:
            cls._logs.clear()

    @classmethod
    def _log_event(cls, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        with cls._lock:
            entry = {
                "timestamp": get_ist_now().strftime("%I:%M:%S %p IST"),
                "type": event_type,
                "message": message,
                "details": details or {}
            }
            cls._logs.insert(0, entry)
            if len(cls._logs) > cls._max_logs:
                cls._logs.pop()

    @classmethod
    def _run_loop(cls):
        """
        Continuous background scanning loop.
        """
        while cls._is_running:
            try:
                now = get_ist_now()
                # Check Intraday Market Hours (09:15 AM - 03:15 PM IST)
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=15, second=0, microsecond=0)
                
                # Check 03:15 PM Auto Square-Off
                if now >= market_close:
                    cls._log_event("MARKET_CLOSE", "3:15 PM IST reached. All intraday positions squared off. Daemon sleeping.")
                    time.sleep(60)
                    continue

                cls._last_scan_time = now.strftime("%I:%M:%S %p IST")
                cls._scans_completed += 1

                # Scan top watchlist stocks
                watchlist = config.DEFAULT_WATCHLIST[:15]
                for item in watchlist:
                    if not cls._is_running:
                        break
                    sym = item["symbol"]
                    cls._evaluate_and_execute_symbol(sym)
                    time.sleep(0.5) # Gentle rate limiting

            except Exception as e:
                cls._log_event("DAEMON_ERROR", f"Error in scan loop: {str(e)}")

            # Sleep until next scan interval
            time.sleep(cls._scan_interval_sec)

    @classmethod
    def _evaluate_and_execute_symbol(cls, symbol: str):
        """
        Evaluates a single symbol through the 2-Stage Council & executes if passed.
        """
        try:
            df = get_historical_data(symbol, period="5d", interval="5m")
            if df.empty or len(df) < 30:
                return

            quote = get_live_quote(symbol)
            
            # 2-Stage Multi-Agent Council Evaluation
            council_res = MultiAgentCouncil.evaluate_candidate(symbol, df, quote)
            
            # If not passed council consensus, do not execute
            if not council_res.get("consensus_approved"):
                return

            blueprint = council_res.get("trade_blueprint", {})
            action = blueprint.get("action", "BUY")
            curr_p = float(council_res.get("current_price", quote.get("price", 100.0)))
            t1 = float(blueprint.get("target_1", {}).get("price", curr_p * 1.025))
            sl = float(blueprint.get("stop_loss", {}).get("price", curr_p * 0.985))
            c_score = float(council_res.get("consensus_score", 8.0))

            # Sizing from available capital (e.g. ₹25,000 max per trade)
            p_state = get_portfolio_state()
            cash = float(p_state.get("cash", 100000.0))
            trade_budget = min(25000.0, cash * 0.25)
            qty = max(1, int(trade_budget / max(1.0, curr_p)))

            proposal = {
                "symbol": symbol,
                "target_asset": symbol,
                "action": "BUY_STOCK" if action == "BUY" else "SELL_STOCK",
                "confidence_score": c_score,
                "entry_price": curr_p,
                "sl": sl,
                "target_1": t1,
                "horizon": "intraday",
                "notes": f"Multi-Agent Council Hunter ({council_res.get('deliberation_summary')})"
            }

            # Guardrail Evaluation
            guard = AIGuardrails(min_confidence_threshold=7.50)
            approved, g_reason, sanitized = guard.evaluate_proposal(proposal, p_state, enforce_time_cutoff=True)

            if not approved:
                cls._log_event("GUARDRAIL_BLOCKED", f"Council approved {display_symbol_name(symbol)} but Guardrail blocked: {g_reason}")
                return

            # Execute Software OCO Entry + SL-M Order
            if cls._broker:
                exec_res = SoftwareOCOManager.execute_guarded_entry_with_oco(
                    broker=cls._broker,
                    symbol=symbol,
                    side=action,
                    quantity=qty,
                    entry_price=curr_p,
                    sl_price=sl,
                    target_1_price=t1,
                    strategy_name="Market_Hunter_Daemon"
                )

                if exec_res.get("status") == "FILLED":
                    cls._trades_placed_today += 1
                    cls._log_event(
                        "TRADE_EXECUTED",
                        f"🚀 EXECUTED {action} {qty}x {display_symbol_name(symbol)} @ ₹{curr_p:,.2f} with SL-M @ ₹{sl:,.2f}.",
                        details=exec_res
                    )
                else:
                    cls._log_event("ORDER_FAILED", f"Order execution failed for {display_symbol_name(symbol)}: {exec_res.get('message')}")

        except Exception as e:
            cls._log_event("SYMBOL_EVAL_ERROR", f"Failed evaluation for {symbol}: {str(e)}")
