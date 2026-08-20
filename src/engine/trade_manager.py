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
        if hasattr(broker, "get_open_positions"):
            active_positions = broker.get_open_positions()
        elif hasattr(broker, "get_positions"):
            active_positions = broker.get_positions()
        else:
            active_positions = get_open_positions()
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
            be_locked = bool(pos.get("breakeven_locked", 0))
            trailing_sl = float(pos.get("trailing_sl") or sl_p)

            # Calculate 1.0R Risk Distance
            initial_risk_r = max(0.01 * entry_p, abs(entry_p - sl_p))

            # -------------------------------------------------------------
            # STAGE 2: Target 1 Milestone (+1.5R) -> Book 50% Profit & Lock +0.5R
            # -------------------------------------------------------------
            if not target_1_hit and side in ["LONG", "BUY"] and curr_p >= t1_p:
                close_qty = max(1, qty // 2) if qty > 1 else qty
                locked_profit_sl = round(entry_p + (0.5 * initial_risk_r), 2)
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

                pos["quantity"] = max(1, qty - close_qty)
                pos["target_1_hit"] = 1
                pos["breakeven_locked"] = 1
                pos["stage"] = "BREAKEVEN_LOCKED"
                pos["sl"] = locked_profit_sl
                pos["trailing_sl"] = locked_profit_sl
                save_position(pos)

                action_events.append({
                    "type": "TARGET_1_PROFIT_BOOKED",
                    "symbol": sym,
                    "price": curr_p,
                    "closed_qty": close_qty,
                    "realized_gain_pct": round(gain_pct, 2),
                    "message": f"Target 1 Hit! Booked 50% profit @ ₹{curr_p:.2f} (+{gain_pct:.1f}%). SL locked to +0.5R (₹{locked_profit_sl:.2f})."
                })
                continue

            # -------------------------------------------------------------
            # STAGE 1: +1.0R Milestone -> Move SL to Breakeven (Entry + Fees)
            # -------------------------------------------------------------
            elif not be_locked and not target_1_hit and side in ["LONG", "BUY"] and curr_p >= (entry_p + initial_risk_r):
                breakeven_sl = round(entry_p * 1.002, 2) # Entry + 0.20% fees buffer
                pos["breakeven_locked"] = 1
                pos["sl"] = max(sl_p, breakeven_sl)
                pos["trailing_sl"] = pos["sl"]
                save_position(pos)
                logger.info(f"🔒 +1.0R Reached on {sym}! Stop-Loss moved to Breakeven (₹{breakeven_sl:.2f}).")
                action_events.append({
                    "type": "BREAKEVEN_LOCKED",
                    "symbol": sym,
                    "price": curr_p,
                    "sl_price": breakeven_sl,
                    "message": f"🔒 +1.0R Milestone hit! Stop-Loss moved to Breakeven @ ₹{breakeven_sl:.2f} (Downside eliminated)."
                })
                continue

            # -------------------------------------------------------------
            # STAGE 3: Target 2 Milestone (+2.5R) -> Full Exit of Runner
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
            # STAGE 4: Dynamic ATR-Adaptive Chandelier Trailing Stop on Runner
            # -------------------------------------------------------------
            if target_1_hit:
                locked_profit_sl = round(entry_p + (0.5 * initial_risk_r), 2)
                # Adaptive Chandelier Trail: Uses instrument ATR if present, else dynamic percentage
                pos_atr = float(pos.get("atr", 0.0))
                if pos_atr > 0:
                    peak_trailing = highest_p - (1.8 * pos_atr)
                else:
                    peak_trailing = highest_p * (1.0 - (trailing_buffer_pct / 100.0))
                
                # Trailing SL never drops below locked +0.5R profit floor
                new_trailing_sl = max(locked_profit_sl, peak_trailing)
                
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
            # STAGE 0/1: Safety Stop-Loss Hit (Before Target 1)
            # -------------------------------------------------------------
            current_active_sl = float(pos.get("sl", sl_p))
            if not target_1_hit and curr_p <= current_active_sl:
                logger.warning(f"🛑 Safety Stop-Loss Hit on {sym} @ ₹{curr_p:.2f}")
                exec_res = broker.square_off_position(sym, reason="Stop-Loss Hit")
                action_events.append({
                    "type": "STOP_LOSS_EXIT",
                    "symbol": sym,
                    "price": curr_p,
                    "closed_qty": qty,
                    "realized_gain_pct": round(gain_pct, 2),
                    "message": f"🛑 Stop-Loss Hit @ ₹{curr_p:.2f} ({gain_pct:.1f}%). Capital preserved."
                })
                continue

            # -------------------------------------------------------------
            # STAGNANT CHOP TIMEOUT CHECK (>45 mins in Chop with <0.25R progress)
            # -------------------------------------------------------------
            entry_time_str = pos.get("entry_time")
            if entry_time_str:
                try:
                    entry_dt = datetime.fromisoformat(str(entry_time_str).replace("Z", "+00:00"))
                    now_dt = datetime.now(timezone.utc) if entry_dt.tzinfo else datetime.now()
                    elapsed_mins = (now_dt - entry_dt).total_seconds() / 60.0
                    r_progress = abs(curr_p - entry_p) / initial_risk_r if initial_risk_r > 0 else 0.0
                    
                    if elapsed_mins >= 45.0 and r_progress <= 0.25 and not target_1_hit:
                        logger.info(f"⏳ Stagnant Chop Timeout on {sym} ({elapsed_mins:.0f}m elapsed, progress {r_progress:.2f}R).")
                        exec_res = broker.square_off_position(sym, reason="Stagnant Chop Timeout (45m Inactivity)")
                        action_events.append({
                            "type": "STAGNANT_CHOP_TIMEOUT",
                            "symbol": sym,
                            "price": curr_p,
                            "closed_qty": qty,
                            "realized_gain_pct": round(gain_pct, 2),
                            "message": f"⏳ Stagnant Position Exited @ ₹{curr_p:.2f} after {elapsed_mins:.0f}m in consolidation. Capital liberated."
                        })
                        continue
                except Exception as e:
                    logger.debug(f"Stagnancy time parse skip: {e}")

            # Save updated position metrics
            save_position(pos)

        return action_events
