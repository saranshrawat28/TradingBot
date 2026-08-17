"""
Deterministic Mathematical Risk Guardrail Layer for AI-Driven Autonomous Trading.
Sits firmly between the AI's trade proposal and the broker execution adapter. Zero bypass.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional
from src.utils.helpers import get_ist_now, is_intraday_squareoff_time

AUDIT_TRAIL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "audit_trail.jsonl")

class AIGuardrails:
    """
    Strict deterministic safety controller protecting trading capital against AI hallucinations,
    over-leveraging, adverse market regimes, and revenge-trading loops.
    """
    
    def __init__(
        self,
        max_daily_loss_flat: float = 2000.0,
        max_daily_loss_pct: float = 3.0,
        max_concurrent_legs: int = 1,
        max_lots_per_trade: int = 1,
        sl_cooldown_minutes: int = 15,
        min_confidence_threshold: float = 7.5,
        max_bid_ask_spread_pct: float = 2.5
    ):
        self.max_daily_loss_flat = max_daily_loss_flat
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_concurrent_legs = max_concurrent_legs
        self.max_lots_per_trade = max_lots_per_trade
        self.sl_cooldown_minutes = sl_cooldown_minutes
        self.min_confidence_threshold = min_confidence_threshold
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct
        
        # State tracking
        self.cooldown_tracker: dict[str, datetime] = {} # symbol -> cooldown expiry time
        self.circuit_broken: bool = False
        self.circuit_break_reason: str = ""

    def evaluate_proposal(
        self,
        proposal: dict,
        portfolio_state: dict,
        current_market_depth: Optional[dict] = None,
        enforce_time_cutoff: bool = False
    ) -> tuple[bool, str, dict]:
        """
        Validate and optionally clamp AI trade proposal against all hard guardrails.
        Returns: (is_approved: bool, reason: str, sanitized_order: dict)
        """
        action = proposal.get("action", "HOLD")
        target_asset = proposal.get("target_asset", "NIFTY")
        confidence = proposal.get("confidence_score", 0.0)
        
        # If AI proposes HOLD or EXIT, allow through safely
        if action == "HOLD":
            return True, "AI proposed HOLD. No action taken.", {"action": "HOLD"}
        if action == "EXIT_POSITION":
            return True, "AI proposed EXIT_POSITION. Square-off approved.", {"action": "EXIT", "symbol": target_asset}
            
        # 1. Circuit Breaker / Daily Loss Limit Check
        daily_pnl = portfolio_state.get("daily_pnl", 0.0)
        total_capital = portfolio_state.get("capital", 100000.0)
        max_loss_allowed = min(self.max_daily_loss_flat, (self.max_daily_loss_pct / 100.0) * total_capital)
        
        if self.circuit_broken or daily_pnl <= -abs(max_loss_allowed):
            self.circuit_broken = True
            self.circuit_break_reason = f"🛑 Hard Daily Drawdown Limit Hit! Realized PnL: ₹{daily_pnl:,.2f} (Limit: -₹{max_loss_allowed:,.2f})"
            self._log_audit("BLOCKED", proposal, self.circuit_break_reason)
            return False, self.circuit_break_reason, {"action": "HOLD"}

        # 2. AI Confidence Threshold Filter
        if confidence < self.min_confidence_threshold:
            reason = f"⚠️ AI Confidence ({confidence}/10) is below required safety threshold ({self.min_confidence_threshold}/10)."
            self._log_audit("BLOCKED", proposal, reason)
            return False, reason, {"action": "HOLD"}
            
        # 3. Post-SL Revenge Trading Cooldown Filter
        now_ist = get_ist_now()
        if target_asset in self.cooldown_tracker:
            cooldown_expiry = self.cooldown_tracker[target_asset]
            if now_ist < cooldown_expiry:
                remaining_sec = int((cooldown_expiry - now_ist).total_seconds())
                reason = f"⏳ Cooldown Active: {target_asset} hit Stop-Loss recently. Locked for {remaining_sec}s to prevent revenge-trading."
                self._log_audit("BLOCKED", proposal, reason)
                return False, reason, {"action": "HOLD"}
            else:
                del self.cooldown_tracker[target_asset]
                
        # 4. Bid-Ask Spread & Liquidity Check
        if current_market_depth:
            bid = float(current_market_depth.get("bid", 0.0))
            ask = float(current_market_depth.get("ask", 0.0))
            ltp = float(current_market_depth.get("price", ask or bid))
            if bid > 0 and ask > 0 and ltp > 0:
                spread_pct = ((ask - bid) / ltp) * 100.0
                if spread_pct > self.max_bid_ask_spread_pct:
                    reason = f"🛑 Illiquid contract rejected: Bid-Ask Spread is {spread_pct:.2f}% (Limit: {self.max_bid_ask_spread_pct}%)."
                    self._log_audit("BLOCKED", proposal, reason)
                    return False, reason, {"action": "HOLD"}

        # 5. Time-of-Day Check (Exclude 09:15-09:25 opening spike & 15:00-15:15 closing auction)
        if enforce_time_cutoff:
            open_spike_cutoff = now_ist.replace(hour=9, minute=25, second=0, microsecond=0)
            market_close_cutoff = now_ist.replace(hour=15, minute=0, second=0, microsecond=0)
            if now_ist < open_spike_cutoff:
                reason = "🛑 Opening Volatility Gate: No new positions permitted during 09:15–09:25 AM opening spike."
                self._log_audit("BLOCKED", proposal, reason)
                return False, reason, {"action": "HOLD"}
            if now_ist >= market_close_cutoff:
                reason = "🛑 Closing Auction Gate: Market is near 3:15 PM IST close. No new positions permitted after 3:00 PM."
                self._log_audit("BLOCKED", proposal, reason)
                return False, reason, {"action": "HOLD"}

        # 6. Index Correlation Gate (Never buy long equity if Nifty is plunging > -1.0%)
        nifty_chg = portfolio_state.get("nifty_change_pct", 0.0)
        is_index = any(idx in target_asset.upper() for idx in ["NIFTY", "BANKNIFTY", "FINNIFTY", "^NSEI", "^NSEBANK"])
        if not is_index and action in ["BUY_STOCK", "BUY_CALL"] and nifty_chg <= -1.0:
            reason = f"🛑 Macro Index Gate: NIFTY 50 is down {nifty_chg:+.2f}%. Fresh long stock entries blocked."
            self._log_audit("BLOCKED", proposal, reason)
            return False, reason, {"action": "HOLD"}

        # 7. Net-of-Fees Risk-to-Reward Ratio Check (Blended Multi-Target Expectancy >= 1.6:1)
        sl_pct = float(proposal.get("suggested_sl_pct", 1.5))
        tp1_pct = float(proposal.get("suggested_tp_pct", 2.25))
        tp2_pct = float(proposal.get("suggested_tp2_pct", tp1_pct * 1.67))
        
        # Blended Gross Target = 50% TP1 + 50% TP2
        blended_tp_pct = (0.5 * tp1_pct) + (0.5 * tp2_pct)
        
        # Deduct estimated statutory round-trip friction (STT, GST, brokerage, slippage ~0.20%)
        net_tp = max(0.1, blended_tp_pct - 0.20)
        net_sl = sl_pct + 0.20
        net_rr = net_tp / net_sl if net_sl > 0 else 0.0
        if net_rr < 1.6:
            reason = f"🛑 Net-of-Fees R:R ratio is too low ({net_rr:.2f}:1 after Indian taxes & fees. Minimum required: 1.6:1)."
            self._log_audit("BLOCKED", proposal, reason)
            return False, reason, {"action": "HOLD"}
                
        # 8. Max Concurrent Position Cap
        open_positions = portfolio_state.get("open_positions", [])
        if len(open_positions) >= self.max_concurrent_legs:
            reason = f"🛑 Max concurrent positions cap reached ({len(open_positions)}/{self.max_concurrent_legs} legs active)."
            self._log_audit("BLOCKED", proposal, reason)
            return False, reason, {"action": "HOLD"}
                    
        # 9. Quantity & Position Sizing Sanitize
        # Calculate standard lot sizes (Nifty = 25, BankNifty = 15, Equities = calculated)
        if "NIFTY" in target_asset:
            lot_size = 25
        elif "BANKNIFTY" in target_asset:
            lot_size = 15
        else:
            lot_size = 1
            
        qty = lot_size * self.max_lots_per_trade
        
        sanitized_order = {
            "action": action,
            "target_asset": target_asset,
            "strike_offset": proposal.get("strike_offset", "ATM"),
            "quantity": qty,
            "suggested_sl_pct": sl_pct,
            "suggested_tp_pct": tp1_pct,
            "suggested_tp2_pct": tp2_pct,
            "confidence_score": confidence,
            "reasoning": proposal.get("reasoning", "")
        }
        
        self._log_audit("APPROVED", proposal, "All guardrails passed successfully.", sanitized_order)
        return True, "Guardrails Passed. Order Approved for Execution.", sanitized_order

    def register_stop_loss_hit(self, symbol: str) -> None:
        """Trigger cooldown timer upon SL execution to prevent revenge trading."""
        cooldown_until = get_ist_now() + timedelta(minutes=self.sl_cooldown_minutes)
        self.cooldown_tracker[symbol] = cooldown_until

    def _log_audit(
        self,
        decision: str,
        raw_proposal: dict,
        reason: str,
        sanitized_order: Optional[dict] = None
    ) -> None:
        """SEBI-compliant immutable audit trail logging."""
        os.makedirs(os.path.dirname(AUDIT_TRAIL_FILE), exist_ok=True)
        entry = {
            "timestamp": get_ist_now().isoformat(),
            "decision": decision,
            "reason": reason,
            "proposal": raw_proposal,
            "sanitized_order": sanitized_order or {}
        }
        with open(AUDIT_TRAIL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
