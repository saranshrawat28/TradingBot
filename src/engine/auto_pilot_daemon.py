"""
Auto-Pilot Background Market Watcher & Continuous Execution Daemon.
Monitors 5-minute candle closes, evaluates AI Radar opportunities, manages dynamic trailing SL,
and enforces 3:15 PM IST square-off autonomously.
"""

import time
import threading
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from src.engine.trade_manager import SmartTradeManager
from src.engine.ai_guardrails import AIGuardrails
from src.engine.reconciliation import StateReconciler
from src.utils.helpers import get_ist_now, is_intraday_squareoff_time

logger = logging.getLogger("AutoPilotDaemon")

class AutoPilotDaemon:
    """
    Background Autonomous Trading Engine.
    """
    
    _instance: Optional["AutoPilotDaemon"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "AutoPilotDaemon":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.last_scan_time: Optional[datetime] = None
        self.last_manage_time: Optional[datetime] = None
        self.scans_count = 0
        self.orders_executed = 0
        self.activity_logs: List[str] = []
        
        # Configuration
        self.llm_client: Any = None
        self.guardrails: Optional[AIGuardrails] = None
        self.broker: Any = None
        self.is_live_mode: bool = False
        self.min_auto_confidence: float = 8.0
        self.candle_interval_sec: int = 300 # 5 minutes
        self.manage_interval_sec: int = 10  # 10 seconds

    def configure(
        self,
        llm_client: Any,
        guardrails: AIGuardrails,
        broker: Any,
        is_live_mode: bool = False,
        min_auto_confidence: float = 8.0
    ):
        """Configure dependencies for the background daemon."""
        self.llm_client = llm_client
        self.guardrails = guardrails
        self.broker = broker
        self.is_live_mode = is_live_mode
        self.min_auto_confidence = min_auto_confidence

    def start(self):
        """Start the background daemon thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._log(f"🟢 Auto-Pilot Engine STARTED | Mode: {'LIVE ZERODHA' if self.is_live_mode else 'PAPER SIMULATION'}")

    def stop(self):
        """Stop the background daemon thread."""
        self.is_running = False
        self._log("⏸️ Auto-Pilot Engine STOPPED by user.")

    def _log(self, message: str):
        now_str = get_ist_now().strftime("%H:%M:%S")
        entry = f"[{now_str}] {message}"
        self.activity_logs.insert(0, entry)
        if len(self.activity_logs) > 100:
            self.activity_logs.pop()
        logger.info(entry)

    def _run_loop(self):
        """Main background loop."""
        last_candle_tick = 0.0

        while self.is_running:
            try:
                now_ist = get_ist_now()

                # 1. Check 3:15 PM IST Auto Square-Off
                if is_intraday_squareoff_time(now_ist):
                    self._log("🛑 3:15 PM IST Reached! Executing Mandatory Intraday Auto-Squareoff...")
                    if self.broker:
                        sq_res = self.broker.square_off_all(reason="3:15 PM Intraday Close")
                        self._log(f"Auto-Squareoff complete: Closed {len(sq_res)} positions.")
                    self.stop()
                    break

                # 2. Position Management & Trailing SL (Every 10 seconds)
                if self.broker:
                    actions = SmartTradeManager.evaluate_and_manage_positions(self.broker)
                    for act in actions:
                        self._log(f"⚡ Trade Event: {act.get('message')}")

                # 3. 5-Minute Candle Scan & AI Opportunity Evaluation
                now_ts = time.time()
                if now_ts - last_candle_tick >= self.candle_interval_sec:
                    last_candle_tick = now_ts
                    self._perform_candle_scan()

            except Exception as e:
                self._log(f"⚠️ Error in background loop: {str(e)}")

            time.sleep(self.manage_interval_sec)

    def _perform_candle_scan(self):
        """Executes a 5-minute market scan and auto-dispatches high conviction setups."""
        if not self.llm_client or not self.llm_client.is_configured():
            return

        from src.ai.market_radar import MarketRadarScanner
        from src.ai.ai_agent import AITradingAgent

        self._log("🔍 5-Minute Candle Close: Scanning Multi-Asset Opportunity Radar...")
        self.last_scan_time = get_ist_now()
        self.scans_count += 1

        radar_res = MarketRadarScanner.scan_market(
            llm_client=self.llm_client,
            min_confidence=self.min_auto_confidence
        )

        if radar_res.get("status") == "SUCCESS":
            opps = radar_res.get("opportunities", [])
            if not opps:
                self._log(f"ℹ️ Market Scan #{self.scans_count}: No setups with confidence >= {self.min_auto_confidence}/10. Preserving capital.")
                return

            top_opp = opps[0]
            conf = float(top_opp.get("confidence_score", 0.0))
            sym = top_opp.get("symbol", "N/A")
            act = top_opp.get("action", "BUY_CALL")
            
            self._log(f"🎯 High-Conviction Signal Detected: {act} on {sym} (Confidence: {conf}/10, Setup: {top_opp.get('setup_name')})")

            # Execute via AI Trading Agent
            agent = AITradingAgent(
                llm_client=self.llm_client,
                guardrails=self.guardrails,
                broker=self.broker,
                is_live_mode=self.is_live_mode
            )
            outcome = agent.execute_radar_opportunity(top_opp)
            if outcome.get("status") == "EXECUTED":
                self.orders_executed += 1
                self._log(f"🚀 ORDER FILLED: {act} {sym} | Target 1: ₹{top_opp.get('target_1')} | SL: ₹{top_opp.get('stop_loss')}")
            else:
                self._log(f"🛡️ Order Blocked by Guardrail: {outcome.get('message')}")
        else:
            self._log(f"⚠️ Scan failed: {radar_res.get('message')}")

    def get_status(self) -> Dict[str, Any]:
        """Fetch current telemetry and status of the daemon."""
        return {
            "is_running": self.is_running,
            "mode": "LIVE ZERODHA" if self.is_live_mode else "PAPER SIMULATION",
            "last_scan_time": self.last_scan_time.strftime("%H:%M:%S IST") if self.last_scan_time else "None",
            "scans_count": self.scans_count,
            "orders_executed": self.orders_executed,
            "logs": self.activity_logs[:20]
        }
