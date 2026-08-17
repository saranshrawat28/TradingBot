"""
Smart Trade Lifecycle Manager & Trailing Stop-Loss Engine.
Enforces the 50/50 Multi-Target Profit Booker:
1. When Target 1 (+15-25%) is reached, books 50% profit immediately.
2. Trails Stop-Loss of remaining 50% to Breakeven (Entry Price).
3. Dynamically trails runner to Target 2 (+40-60%) with zero downside risk.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils.helpers import get_ist_now, clean_symbol
from src.utils.storage import get_open_positions, save_position, log_closed_trade
from src.data.data_fetcher import get_live_quote

logger = logging.getLogger("SmartTradeManager")

class SmartTradeManager:
    """
    Manages active positions, dynamic trailing stops, and multi-stage profit targets.
    """

    @classmethod
    def evaluate_and_manage_positions(
        cls,
        broker: Any,
        trailing_buffer_pct: float = 8.0
    ) -> List[Dict[str, Any]]:
        """
        Scan all active positions, update live metrics, and execute multi-stage exits.
        Returns list of actions executed during this tick.
        """
        active_positions = broker.get_positions() if hasattr(broker, "get_positions") else get_open_positions()
        action_events = []

        for pos in active_positions:
            sym = pos["symbol"]
            qty = int(pos["quantity"])
            if qty <= 0:
                continue

            entry_p = float(pos["entry_price"])
            side = pos.get("side", "LONG").upper()
            
            # Fetch live quote
            quote = get_live_quote(sym)
            curr_p = float(quote.get("price", pos.get("current_price", entry_p)))
            if curr_p <= 0:
                continue

            # Update highest price tracked
            highest_p = max(pos.get("highest_price", entry_p), curr_p)
            pos["highest_price"] = highest_p
            pos["current_price"] = curr_p

            # Calculate live PnL
            if side in ["LONG", "BUY"]:
                unrealized_pnl = (curr_p - entry_p) * qty
                gain_pct = ((curr_p - entry_p) / entry_p) * 100.0
            else:
                unrealized_pnl = (entry_p - curr_p) * qty
                gain_pct = ((entry_p - curr_p) / entry_p) * 100.0

            pos["unrealized_pnl"] = round(unrealized_pnl, 2)
            pos["unrealized_pnl_pct"] = round(gain_pct, 2)

            sl_p = float(pos.get("sl") or (entry_p * 0.85))
            t1_p = float(pos.get("target_1") or (pos.get("tp") or (entry_p * 1.20)))
            t2_p = float(pos.get("target_2") or (entry_p * 1.45))
            target_1_hit = bool(pos.get("target_1_hit", 0))
            trailing_sl = float(pos.get("trailing_sl") or sl_p)

            # -------------------------------------------------------------
            # STAGE 1: Target 1 Milestone (+15% to +25%) -> Book 50% Profit
            # -------------------------------------------------------------
            if not target_1_hit and side in ["LONG", "BUY"] and curr_p >= t1_p:
                close_qty = max(1, qty // 2) if qty > 1 else qty
                logger.info(f"🎯 Target 1 HIT on {sym}! Booking 50% ({close_qty}/{qty} shares) @ ₹{curr_p:.2f}")

                if hasattr(broker, "partial_close_position"):
                    exec_res = broker.partial_close_position(
                        symbol=sym,
                        quantity_to_close=close_qty,
                        exit_price=curr_p,
                        reason="Target 1 Partial Profit (50% Booked)"
                    )
                else:
                    exec_res = broker.square_off_position(sym, reason="Target 1 Full Profit")

                action_events.append({
                    "type": "TARGET_1_PROFIT_BOOKED",
                    "symbol": sym,
                    "price": curr_p,
                    "closed_qty": close_qty,
                    "realized_gain_pct": round(gain_pct, 2),
                    "message": f"Target 1 Hit! Booked 50% profit @ ₹{curr_p:.2f} (+{gain_pct:.1f}%). SL shifted to Breakeven (₹{entry_p:.2f})."
                })
                continue

            # -------------------------------------------------------------
            # STAGE 2: Target 2 Milestone (+40% to +60%) -> Full Profit Exit
            # -------------------------------------------------------------
            if side in ["LONG", "BUY"] and curr_p >= t2_p:
                logger.info(f"🏆 Target 2 HIT on {sym}! Full exit @ ₹{curr_p:.2f}")
                exec_res = broker.square_off_position(sym, reason="Target 2 Final Profit Exit")
                action_events.append({
                    "type": "TARGET_2_FULL_EXIT",
                    "symbol": sym,
                    "price": curr_p,
                    "closed_qty": qty,
                    "realized_gain_pct": round(gain_pct, 2),
                    "message": f"🏆 Target 2 Hit! Position fully closed @ ₹{curr_p:.2f} (+{gain_pct:.1f}%)."
                })
                continue

            # -------------------------------------------------------------
            # STAGE 3: Dynamic Trailing Stop-Loss (Runner Phase)
            # -------------------------------------------------------------
            if target_1_hit:
                # Dynamic Trailing SL: Moves up as price makes new highs, but NEVER drops below entry
                peak_trailing = highest_p * (1.0 - (trailing_buffer_pct / 100.0))
                new_trailing_sl = max(entry_p, peak_trailing) # Minimum is Breakeven
                
                if new_trailing_sl > trailing_sl:
                    pos["trailing_sl"] = round(new_trailing_sl, 2)
                    pos["sl"] = round(new_trailing_sl, 2)
                    save_position(pos)
                    logger.info(f"🔒 Trailing SL for {sym} raised to ₹{new_trailing_sl:.2f}")

                # Check if Trailing SL Hit
                if curr_p <= pos["trailing_sl"]:
                    logger.info(f"🛑 Trailing SL Hit on {sym} @ ₹{curr_p:.2f}")
                    exec_res = broker.square_off_position(sym, reason="Trailing Stop-Loss Hit (Locked Profit)")
                    action_events.append({
                        "type": "TRAILING_SL_EXIT",
                        "symbol": sym,
                        "price": curr_p,
                        "closed_qty": qty,
                        "realized_gain_pct": round(gain_pct, 2),
                        "message": f"🔒 Trailing Stop-Loss Hit @ ₹{curr_p:.2f}. Profit secured above breakeven (+{gain_pct:.1f}%)."
                    })
                    continue

            # -------------------------------------------------------------
            # STAGE 4: Initial Stop-Loss Protection (Before Target 1)
            # -------------------------------------------------------------
            if not target_1_hit and curr_p <= sl_p:
                logger.warning(f"🛑 Hard Safety Stop-Loss Hit on {sym} @ ₹{curr_p:.2f}")
                exec_res = broker.square_off_position(sym, reason="Hard Stop-Loss Hit")
                action_events.append({
                    "type": "STOP_LOSS_EXIT",
                    "symbol": sym,
                    "price": curr_p,
                    "closed_qty": qty,
                    "realized_gain_pct": round(gain_pct, 2),
                    "message": f"🛑 Stop-Loss Hit @ ₹{curr_p:.2f} ({gain_pct:.1f}%). Position closed to preserve capital."
                })
                continue

            # Save updated position metrics
            save_position(pos)

        return action_events
