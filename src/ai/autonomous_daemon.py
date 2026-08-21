"""
Autonomous AI Trading Bot Daemon.
Runs continuous background market scans, evaluates high-probability setups,
coordinates 3-Agent Council verification, triggers autonomous order execution,
and streams real-time AI thought reasoning into the dashboard.
"""

import time
import threading
import logging
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.ai.market_radar import MarketRadarScanner
from src.ai.ai_agent import AITradingAgent
from src.engine.ai_guardrails import AIGuardrails
from src.engine.trade_manager import SmartTradeManager
from src.utils.helpers import get_ist_now

logger = logging.getLogger("AutonomousAIDaemon")

class AutonomousAIDaemon:
    """
    Autonomous AI trading background engine.
    Continuously monitors watchlist, logs internal reasoning, and executes approved trades.
    """
    
    _instance: Optional["AutonomousAIDaemon"] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> "AutonomousAIDaemon":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
            
    def __init__(self):
        self.is_active = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.scan_interval = 20  # seconds between scans
        self.min_confidence = 8.0 # Auto-pilot minimum confidence threshold
        self.thought_stream = deque(maxlen=60) # Store last 60 thought logs
        self.llm_client = None
        self.guardrails = None
        self.broker = None
        self.is_live_mode = False
        self.last_scan_time: Optional[datetime] = None
        self.trades_executed_today = 0
        self._add_thought("SYSTEM", "AI Autonomous Trading Daemon initialized and ready.", symbol="SYSTEM", conviction=10.0)

    def _add_thought(self, level: str, message: str, symbol: str = "SCANNER", conviction: float = 0.0):
        """Thread-safe append to the in-memory AI Thought Stream."""
        ist_str = get_ist_now().strftime("%H:%M:%S")
        entry = {
            "time": ist_str,
            "level": level,  # 'SCAN', 'SETUP', 'EXECUTE', 'RISK', 'EXIT', 'SYSTEM'
            "symbol": symbol,
            "message": message,
            "conviction": conviction
        }
        self.thought_stream.appendleft(entry)
        logger.info(f"[{ist_str}] [{level}] {symbol}: {message}")

    def get_thought_stream(self) -> List[Dict[str, Any]]:
        """Returns snapshot of recent thoughts."""
        return list(self.thought_stream)

    def start(self, llm_client: Any, guardrails: AIGuardrails, broker: Any, is_live_mode: bool = False, interval: int = 20):
        """Starts the autonomous daemon thread."""
        with self._lock:
            if self.is_active and self._thread and self._thread.is_alive():
                return
            
            self.llm_client = llm_client
            self.guardrails = guardrails
            self.broker = broker
            self.is_live_mode = is_live_mode
            self.scan_interval = max(5, interval)
            self._stop_event.clear()
            self.is_active = True
            
            self._thread = threading.Thread(target=self._run_loop, name="AutonomousAIDaemonWorker", daemon=True)
            self._thread.start()
            self._add_thought("SYSTEM", f"🚀 Auto-Pilot Daemon STARTED (Scan Interval: {self.scan_interval}s, Mode: {'LIVE' if is_live_mode else 'PAPER'})", symbol="DAEMON", conviction=10.0)

    def stop(self):
        """Stops the autonomous daemon thread gracefully."""
        with self._lock:
            if not self.is_active:
                return
            self._stop_event.set()
            self.is_active = False
            self._add_thought("SYSTEM", "🛑 Auto-Pilot Daemon STOPPED by user.", symbol="DAEMON", conviction=10.0)

    def _run_loop(self):
        """Main background loop."""
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                self._execute_cycle()
            except Exception as e:
                logger.error(f"Error in autonomous daemon cycle: {e}", exc_info=True)
                self._add_thought("RISK", f"Scan cycle error: {str(e)[:120]}", symbol="ERROR")
                
            elapsed = time.time() - loop_start
            sleep_time = max(1.0, self.scan_interval - elapsed)
            
            # Sleep in short slices to respond to stop signal immediately
            slices = int(sleep_time / 0.5)
            for _ in range(max(1, slices)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.5)

    def _execute_cycle(self):
        """Single autonomous scanning, trade evaluation, and management cycle."""
        now = get_ist_now()
        self.last_scan_time = now
        
        # 1. Check Mandatory Intraday 3:15 PM Exit
        if now.hour == 15 and now.minute >= 15:
            self._add_thought("EXIT", "⏰ 03:15 PM IST Reached: Triggering mandatory end-of-day square-off protocol.", symbol="TIME_GATE")
            if self.broker:
                open_pos = self.broker.get_positions()
                if open_pos:
                    for pos in open_pos:
                        sym = pos.get("symbol")
                        self.broker.square_off_position(sym)
                        self._add_thought("EXIT", f"Auto-squared off {sym} at market close.", symbol=sym)
            return

        # 2. Position Management: Manage active open positions with SmartTradeManager
        if self.broker:
            try:
                positions = self.broker.get_positions()
                if positions:
                    for pos in positions:
                        sym = pos.get("symbol")
                        res = SmartTradeManager.evaluate_and_manage_position(pos, self.broker)
                        if res.get("action") in ["TARGET_1_PARTIAL_EXIT", "TARGET_2_FULL_EXIT", "STOP_LOSS_EXIT", "TRAILING_STOP_EXIT"]:
                            self._add_thought("MANAGEMENT", f"SmartTradeManager: {res.get('message')}", symbol=sym)
            except Exception as e:
                logger.warning(f"Position management check error: {e}")

        # 3. Market Scan for New Opportunities
        self._add_thought("SCAN", f"Scanning NSE/NFO watchlist for high-probability institutional setups...", symbol="SCANNER")
        radar_result = MarketRadarScanner.scan_market(llm_client=self.llm_client, min_confidence=7.0, force_refresh=True)
        
        if radar_result.get("status") != "SUCCESS":
            self._add_thought("SCAN", f"Scan completed: No setups found or data unavailable.", symbol="SCANNER")
            return
            
        opps = radar_result.get("opportunities", [])
        if not opps:
            self._add_thought("SCAN", f"Watchlist evaluated: Market in consolidation, 0 setups meeting >= 7.0 confidence.", symbol="SCANNER")
            return

        # 4. Filter for Top Autonomous Candidate
        top_opp = opps[0]
        sym = top_opp.get("symbol", "N/A")
        contract = top_opp.get("option_contract", sym)
        conf = float(top_opp.get("confidence_score", 0.0))
        act = top_opp.get("action", "BUY_CALL")
        setup = top_opp.get("setup_name", "Momentum Setup")
        entry_p = float(top_opp.get("entry_price", 0.0))
        
        self._add_thought(
            "SETUP",
            f"Found Setup: #{top_opp.get('rank', 1)} {contract} ({act}) | Conviction: {conf:.1f}/10 | Setup: {setup}",
            symbol=sym,
            conviction=conf
        )

        # 5. Check Autonomous Execution Threshold (min_confidence >= 8.0)
        if conf >= self.min_confidence:
            self._add_thought(
                "EXECUTE",
                f"⚡ Conviction {conf:.1f} >= {self.min_confidence} threshold! Initiating AITradingAgent execution pipeline...",
                symbol=sym,
                conviction=conf
            )
            
            # Build agent and execute
            agent = AITradingAgent(
                llm_client=self.llm_client,
                guardrails=self.guardrails or AIGuardrails(),
                broker=self.broker,
                is_live_mode=self.is_live_mode
            )
            
            outcome = agent.execute_radar_opportunity(top_opp)
            if outcome.get("status") == "EXECUTED":
                self.trades_executed_today += 1
                self._add_thought(
                    "EXECUTE",
                    f"✅ TRADE EXECUTED AUTONOMOUSLY: {contract} @ ₹{entry_p:,.2f} | Order ID: {outcome.get('order_id', 'AUTO')}",
                    symbol=sym,
                    conviction=conf
                )
            else:
                self._add_thought(
                    "RISK",
                    f"🛡️ Execution Guardrail Veto: {outcome.get('message')}",
                    symbol=sym,
                    conviction=conf
                )
        else:
            self._add_thought(
                "SCAN",
                f"Setup {contract} conviction {conf:.1f}/10 below auto-pilot {self.min_confidence} bar. Stored on Radar for manual 1-click.",
                symbol=sym,
                conviction=conf
            )
