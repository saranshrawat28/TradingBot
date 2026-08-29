"""
Intraday Telemetry Tracker for Paper Lab.
Polls every 30 minutes during market hours for UI visualization and stale feed monitoring.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from src.paper_lab.paper_db import PaperDB
from src.data.data_fetcher import get_live_quote
from src.utils.helpers import get_ist_now

class LiveTracker:
    """Polls live prices for active paper picks and logs UI snapshots."""

    @classmethod
    def track_active_picks(cls, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Polls live prices for all active picks and records snapshots.
        """
        now = get_ist_now()
        date_str = target_date or now.strftime("%Y-%m-%d")

        picks = PaperDB.get_picks_by_date(date_str)
        if not picks:
            return []

        snapshots = []
        now_str = now.strftime("%H:%M:%S")

        for pick in picks:
            symbol = pick["symbol"]
            entry_p = float(pick.get("entry_price") or pick.get("signal_price", 100.0))
            qty = int(pick.get("quantity", 1))

            quote = get_live_quote(symbol)
            curr_p = float(quote.get("price", entry_p))
            if curr_p <= 0:
                curr_p = entry_p

            unrealized_rs = round((curr_p - entry_p) * qty, 2)
            unrealized_pct = round(((curr_p - entry_p) / entry_p) * 100.0, 2)

            t1_p = float(pick["target_1"])
            t2_p = float(pick["target_2"])
            sl_p = float(pick["stop_loss"])

            if curr_p >= t2_p:
                status = "TARGET_2_REACHED"
            elif curr_p >= t1_p:
                status = "TARGET_1_REACHED"
            elif curr_p <= sl_p:
                status = "STOP_LOSS_REACHED"
            else:
                status = "IN_ZONE"

            snap = {
                "pick_date": date_str,
                "symbol": symbol,
                "timestamp": now_str,
                "current_price": curr_p,
                "unrealized_pnl_rs": unrealized_rs,
                "unrealized_pnl_pct": unrealized_pct,
                "status": status
            }
            PaperDB.save_snapshot(snap)
            snapshots.append(snap)

        return snapshots
