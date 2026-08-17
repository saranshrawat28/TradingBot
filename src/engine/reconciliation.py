"""
Broker State Reconciliation & Startup Crash Recovery Engine.
Ensures the exchange/broker is always the single source of ground truth on restarts and reconnects.
"""

import logging
from typing import Any
from src.utils.storage import get_portfolio_state, get_open_positions, log_order

logger = logging.getLogger("Reconciliation")

class StateReconciler:
    """
    Synchronizes local bot state with actual broker positions on startup or network reconnect.
    """
    
    @staticmethod
    def reconcile_with_broker(broker: Any) -> dict:
        """
        Reconcile local state with broker's live exchange state.
        Returns reconciled portfolio snapshot.
        """
        try:
            # 1. Fetch live ground truth positions directly from broker API
            broker_positions = broker.get_positions()
            broker_margins = broker.get_margins()
            
            # 2. Extract active open legs
            reconciled_open = []
            for p in broker_positions:
                qty = int(p.get("quantity", 0))
                if qty != 0:
                    reconciled_open.append({
                        "symbol": p.get("symbol") or p.get("tradingsymbol"),
                        "quantity": abs(qty),
                        "side": "BUY" if qty > 0 else "SELL",
                        "entry_price": float(p.get("average_price") or p.get("buy_price") or p.get("price", 0.0)),
                        "current_price": float(p.get("last_price") or p.get("close_price") or 0.0),
                        "pnl": float(p.get("pnl") or p.get("unrealised", 0.0)),
                        "product": p.get("product", "MIS")
                    })
                    
            available_cash = float(broker_margins.get("available_cash") or broker_margins.get("equity", {}).get("available", {}).get("cash", 100000.0))
            
            # Calculate total day's realized PnL from broker
            total_realized_pnl = sum(float(p.get("m2m", 0.0)) for p in broker_positions if int(p.get("quantity", 0)) == 0)
            
            reconciled_state = {
                "status": "SYNCED",
                "capital": available_cash,
                "daily_pnl": total_realized_pnl,
                "open_positions": reconciled_open,
                "active_legs_count": len(reconciled_open),
                "broker_name": broker.__class__.__name__
            }
            return reconciled_state
            
        except Exception as e:
            logger.error(f"Reconciliation error: {str(e)}")
            # Fallback to local storage if broker connection times out
            local_positions = get_open_positions()
            local_portfolio = get_portfolio_state()
            return {
                "status": "LOCAL_FALLBACK",
                "capital": local_portfolio.get("capital", 100000.0),
                "daily_pnl": local_portfolio.get("daily_pnl", 0.0),
                "open_positions": local_positions,
                "active_legs_count": len(local_positions),
                "error": str(e)
            }
