"""
DhanHQ Broker Adapter for Indian Stock Market.
"""

import config
from src.brokers.base_broker import BaseBroker

class DhanBroker(BaseBroker):
    """
    DhanHQ API connector.
    """
    
    def __init__(self, client_id: str = None, access_token: str = None):
        super().__init__(name="DhanHQ")
        self.client_id = client_id or config.DHAN_CLIENT_ID
        self.access_token = access_token or config.DHAN_ACCESS_TOKEN
        self.dhan = None
        self.connect()

    def connect(self) -> bool:
        if not self.client_id or not self.access_token:
            self.is_connected = False
            return False
            
        try:
            from dhanhq import dhanhq
            self.dhan = dhanhq(self.client_id, self.access_token)
            fund_limit = self.dhan.get_fund_limits()
            if fund_limit.get("status") == "success":
                self.is_connected = True
                return True
            else:
                self.is_connected = False
                return False
        except Exception as e:
            print(f"DhanHQ notice: {e}")
            self.is_connected = False
            return False

    def get_account_balance(self) -> dict:
        if not self.is_connected or not self.dhan:
            return {"cash": 0.0, "margin_used": 0.0, "status": "Not Connected"}
        try:
            limits = self.dhan.get_fund_limits()
            data = limits.get("data", {})
            avail = float(data.get("availabelBalance", 0.0))
            return {"cash": avail, "margin_used": 0.0, "status": "Connected"}
        except Exception as e:
            return {"cash": 0.0, "error": str(e)}

    def get_open_positions(self) -> list[dict]:
        if not self.is_connected or not self.dhan:
            return []
        try:
            res = self.dhan.get_positions()
            data = res.get("data", []) or []
            return [
                {
                    "symbol": p.get("tradingSymbol"),
                    "side": "LONG" if int(p.get("netQty", 0)) > 0 else "SHORT",
                    "quantity": abs(int(p.get("netQty", 0))),
                    "entry_price": float(p.get("buyAvg", 0.0)),
                    "current_price": float(p.get("costPrice", 0.0)),
                    "unrealized_pnl": float(p.get("realizedProfit", 0.0))
                }
                for p in data if int(p.get("netQty", 0)) != 0
            ]
        except Exception as e:
            return []

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float = None,
        order_type: str = "MARKET",
        product: str = "INTRADAY",
        sl: float = None,
        tp: float = None,
        strategy_name: str = "Manual"
    ) -> dict:
        if not self.is_connected or not self.dhan:
            return {"status": "REJECTED", "message": "DhanHQ is not connected"}
        return {"status": "REJECTED", "message": "Live order execution requires verified live API session."}

    def square_off_position(self, symbol: str, reason: str = "MANUAL") -> dict:
        return {"status": "FAILED", "message": "Not implemented in offline demo mode"}

    def square_off_all(self, reason: str = "MANUAL") -> list[dict]:
        return []
