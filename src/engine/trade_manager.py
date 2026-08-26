"""
Smart Trade Lifecycle Manager & Dynamic ATR Trailing Stop Ratchet Engine.
Enforces the Institutional 4-Stage Profit Locker:
1. Stage 1 (Initial): Capped 1.0R risk floor.
2. Stage 2 (+1.0R): Ratchets Stop-Loss to Breakeven (+0.20% fees buffer) -> 100% Risk Free.
3. Stage 3 (Target 1 Hit, +1.5R): Books 50% partial profit, locks +0.75R profit floor, trails remaining 50% runner using Adaptive Chandelier ATR (1.5x ATR).
4. Stage 4 (Parabolic Runner > +3.0R): Tightens trail to 1.0x ATR, locking super-trends up to +10R+.
5. Dynamic Broker Sync: Modifies active exchange SL-M orders on Zerodha / Dhan as the ratchet climbs.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from src.utils.helpers import get_ist_now, clean_symbol
from src.utils.storage import get_open_positions, save_position, log_closed_trade
from src.data.data_fetcher import get_live_quote, get_historical_data
from src.strategies.indicators import calculate_atr, calculate_trailing_ratchet_levels
from src.engine.software_oco_manager import SoftwareOCOManager

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

            # Update highest/lowest price tracked
            highest_p = max(pos.get("highest_price", entry_p), curr_p)
            lowest_p = min(pos.get("lowest_price", entry_p), curr_p)
            pos["highest_price"] = highest_p
            pos["lowest_price"] = lowest_p
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

            sl_p = float(pos.get("sl") or (entry_p * 0.85 if side in ["LONG", "BUY"] else entry_p * 1.15))
            t1_p = float(pos.get("target_1") or (pos.get("tp") or (entry_p * 1.03 if side in ["LONG", "BUY"] else entry_p * 0.97)))
            t2_p = float(pos.get("target_2") or (entry_p * 1.06 if side in ["LONG", "BUY"] else entry_p * 0.94))
            target_1_hit = bool(pos.get("target_1_hit", 0))
            be_locked = bool(pos.get("breakeven_locked", 0))
            trailing_sl = float(pos.get("trailing_sl") or sl_p)

            # Calculate 1.0R Risk Distance
            initial_risk_r = float(pos.get("initial_risk_r", 0.0))
            if initial_risk_r <= 0:
                initial_risk_r = max(0.005 * entry_p, abs(entry_p - sl_p))
                pos["initial_risk_r"] = initial_risk_r

            # Fetch or calculate ATR for dynamic chandelier trail
            pos_atr = float(pos.get("atr", 0.0))
            if pos_atr <= 0:
                try:
                    df = get_historical_data(sym, period="5d", interval="15m")
                    if df is not None and len(df) >= 14:
                        atr_series = calculate_atr(df["High"], df["Low"], df["Close"], period=14)
                        pos_atr = float(atr_series.iloc[-1])
                        pos["atr"] = round(pos_atr, 2)
                except Exception:
                    pos_atr = entry_p * 0.015

            # -------------------------------------------------------------
            # STAGE 2: Target 1 Milestone (+1.5R) -> Book 50% Profit & Lock Floor
            # -------------------------------------------------------------
            t1_reached = (curr_p >= t1_p) if side in ["LONG", "BUY"] else (curr_p <= t1_p)
            if not target_1_hit and t1_reached:
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

                pos["quantity"] = max(1, qty - close_qty)
                pos["target_1_hit"] = 1
                pos["breakeven_locked"] = 1
                pos["stage"] = "T1_BOOKED_RUNNER_TRAILING"

                # Calculate fresh ratchet levels with T1 active
                ratchet = calculate_trailing_ratchet_levels(
                    entry_price=entry_p,
                    highest_price=highest_p,
                    current_price=curr_p,
                    atr_val=pos_atr,
                    initial_risk_r=initial_risk_r,
                    side=side,
                    target_1_hit=True,
                    current_trailing_sl=trailing_sl
                )
                pos["sl"] = ratchet["new_sl"]
                pos["trailing_sl"] = ratchet["new_sl"]
                pos["locked_r"] = ratchet["locked_r"]
                pos["stage"] = ratchet["stage"]

                # Modify exchange SL-M order if open
                if pos.get("sl_order_id"):
                    SoftwareOCOManager.modify_exchange_sl_order(broker, sym, ratchet["new_sl"], pos.get("sl_order_id"))

                save_position(pos)

                action_events.append({
                    "type": "TARGET_1_PROFIT_BOOKED",
                    "symbol": sym,
                    "price": curr_p,
                    "closed_qty": close_qty,
                    "realized_gain_pct": round(gain_pct, 2),
                    "locked_sl": ratchet["new_sl"],
                    "locked_r": ratchet["locked_r"],
                    "message": f"Target 1 Hit! Booked 50% profit @ ₹{curr_p:.2f} (+{gain_pct:.1f}%). Runner SL locked to +{ratchet['locked_r']}R (₹{ratchet['new_sl']:.2f})."
                })
                continue

            # -------------------------------------------------------------
            # STAGE 2.5: Target 2 Hit -> Full Profit Exit on Runner
            # -------------------------------------------------------------
            t2_reached = (curr_p >= t2_p) if side in ["LONG", "BUY"] else (curr_p <= t2_p)
            if target_1_hit and t2_reached:
                logger.info(f"🏆 Target 2 HIT on {sym}! Full Profit Exit @ ₹{curr_p:.2f}")
                exec_res = broker.square_off_position(sym, reason="Target 2 Hit (Full Profit Exit)")
                action_events.append({
                    "type": "TARGET_2_FULL_EXIT",
                    "symbol": sym,
                    "price": curr_p,
                    "closed_qty": qty,
                    "realized_gain_pct": round(gain_pct, 2),
                    "message": f"🏆 Target 2 Hit @ ₹{curr_p:.2f} (+{gain_pct:.1f}%). Full position closed with maximum gain!"
                })
                continue

            # -------------------------------------------------------------
            # DYNAMIC ATR TRAILING RATCHET EVALUATION (All Active Stages)
            # -------------------------------------------------------------
            ratchet = calculate_trailing_ratchet_levels(
                entry_price=entry_p,
                highest_price=highest_p,
                current_price=curr_p,
                atr_val=pos_atr,
                initial_risk_r=initial_risk_r,
                side=side,
                target_1_hit=target_1_hit,
                current_trailing_sl=trailing_sl
            )

            # Update position metrics if ratcheted higher
            if ratchet["new_sl"] != trailing_sl:
                prev_sl = trailing_sl
                pos["trailing_sl"] = ratchet["new_sl"]
                pos["sl"] = ratchet["new_sl"]
                pos["stage"] = ratchet["stage"]
                pos["locked_r"] = ratchet["locked_r"]

                # Sync modified Stop-Loss to Exchange
                if pos.get("sl_order_id"):
                    SoftwareOCOManager.modify_exchange_sl_order(broker, sym, ratchet["new_sl"], pos.get("sl_order_id"))

                save_position(pos)

                if ratchet["stage"] == "BREAKEVEN_LOCKED" and not be_locked:
                    pos["breakeven_locked"] = 1
                    logger.info(f"🔒 +1.0R Reached on {sym}! Stop-Loss moved to Breakeven (₹{ratchet['new_sl']:.2f}).")
                    action_events.append({
                        "type": "BREAKEVEN_LOCKED",
                        "symbol": sym,
                        "price": curr_p,
                        "sl_price": ratchet["new_sl"],
                        "message": f"🔒 +1.0R Milestone hit! Stop-Loss moved to Breakeven @ ₹{ratchet['new_sl']:.2f} (Downside eliminated)."
                    })
                elif ratchet["new_sl"] > prev_sl and target_1_hit:
                    logger.info(f"🚀 Trailing Ratchet on {sym} raised to ₹{ratchet['new_sl']:.2f} (Locked: +{ratchet['locked_r']}R | Stage: {ratchet['stage']})")

            # -------------------------------------------------------------
            # EXIT CHECK 1: Trailing SL Hit on Runner (Guaranteed Profit)
            # -------------------------------------------------------------
            active_trailing_sl = float(pos.get("trailing_sl", sl_p))
            trailing_hit = (curr_p <= active_trailing_sl) if side in ["LONG", "BUY"] else (curr_p >= active_trailing_sl)

            if target_1_hit and trailing_hit:
                logger.info(f"🛑 Trailing SL Hit on Runner {sym} @ ₹{curr_p:.2f} (Floor: ₹{active_trailing_sl:.2f})")
                exec_res = broker.square_off_position(sym, reason=f"Trailing Stop-Loss Hit (Locked +{pos.get('locked_r', 0.5)}R Profit)")
                action_events.append({
                    "type": "TRAILING_SL_EXIT",
                    "symbol": sym,
                    "price": curr_p,
                    "closed_qty": qty,
                    "realized_gain_pct": round(gain_pct, 2),
                    "locked_r": pos.get("locked_r", 0.75),
                    "message": f"🔒 Trailing Stop-Loss Hit @ ₹{curr_p:.2f}. Runner profit secured (+{gain_pct:.1f}% / +{pos.get('locked_r', 0.75)}R)."
                })
                continue

            # -------------------------------------------------------------
            # EXIT CHECK 2: Safety Stop-Loss Hit (Before Target 1)
            # -------------------------------------------------------------
            if not target_1_hit and trailing_hit:
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

