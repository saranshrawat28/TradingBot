"""
Angel One SmartAPI Broker Adapter.
"""

import config
from src.brokers.base_broker import BaseBroker

class AngelOneBroker(BaseBroker):
    """
    Angel One SmartAPI connector.
    Supports TOTP 2FA authentication, order placement, and position tracking.
    """
    
    def __init__(self, api_key: str = None, client_id: str = None, pin: str = None, totp_key: str = None):
        super().__init__(name="Angel One SmartAPI")
        self.api_key = api_key or config.ANGEL_API_KEY
        self.client_id = client_id or config.ANGEL_CLIENT_ID
        self.pin = pin or config.ANGEL_PIN
        self.totp_key = totp_key or config.ANGEL_TOTP_KEY
        self.smart_api = None
        self.connect()

    def connect(self) -> bool:
        if not self.api_key or not self.client_id or not self.pin:
            self.is_connected = False
            return False
            
        try:
            # We can import SmartConnect if available
            from smartapi import SmartConnect
            import pyotp
            
            self.smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_key).now() if self.totp_key else ""
            data = self.smart_api.generateSession(self.client_id, self.pin, totp)
            if data.get("status"):
                self.is_connected = True
                return True
            else:
                self.is_connected = False
                return False
        except Exception as e:
            print(f"Angel One notice: {e}")
            self.is_connected = False
            return False

    def get_account_balance(self) -> dict:
        if not self.is_connected or not self.smart_api:
            return {"cash": 0.0, "margin_used": 0.0, "status": "Not Connected"}
        try:
            rms = self.smart_api.rmsLimit()
            cash = float(rms.get("data", {}).get("net", 0.0))
            return {"cash": cash, "margin_used": 0.0, "status": "Connected"}
        except Exception as e:
            return {"cash": 0.0, "error": str(e)}

    def get_open_positions(self) -> list[dict]:
        if not self.is_connected or not self.smart_api:
            return []
        try:
            pos = self.smart_api.position()
            positions_data = pos.get("data", []) or []
            return [
                {
                    "symbol": p.get("tradingsymbol"),
                    "side": "LONG" if int(p.get("netqty", 0)) > 0 else "SHORT",
                    "quantity": abs(int(p.get("netqty", 0))),
                    "entry_price": float(p.get("avgprice", 0.0)),
                    "current_price": float(p.get("ltp", 0.0)),
                    "unrealized_pnl": float(p.get("pnl", 0.0))
                }
                for p in positions_data if int(p.get("netqty", 0)) != 0
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
        if not self.is_connected or not self.smart_api:
            return {"status": "REJECTED", "message": "Angel One is not connected"}
        try:
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol.replace(".NS", ""),
                "symboltoken": "3045", # Sample or resolved token
                "transactiontype": side.upper(),
                "exchange": "NSE",
                "ordertype": order_type,
                "producttype": "INTRADAY" if product == "MIS" else "DELIVERY",
                "duration": "DAY",
                "price": str(price) if price else "0",
                "quantity": str(quantity)
            }
            order_id = self.smart_api.placeOrder(order_params)
            return {"status": "FILLED", "order_id": order_id, "symbol": symbol}
        except Exception as e:
            return {"status": "FAILED", "message": str(e)}

    def square_off_position(self, symbol: str, reason: str = "MANUAL") -> dict:
        return {"status": "FAILED", "message": "Not implemented in offline demo mode"}

    def square_off_all(self, reason: str = "MANUAL") -> list[dict]:
        return []
