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

    def get_margins(self) -> dict:
        """Fetch available margin for state reconciliation."""
        if not self.is_connected or not self.smart_api:
            return {"available_cash": 100000.0, "used_margin": 0.0}
        try:
            rms = self.smart_api.rmsLimit()
            cash = float(rms.get("data", {}).get("net", 100000.0))
            return {"available_cash": cash, "used_margin": 0.0}
        except Exception as e:
            return {"available_cash": 100000.0, "used_margin": 0.0}

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
                    "unrealized_pnl": float(p.get("pnl", 0.0)),
                    "product": p.get("producttype", "INTRADAY")
                }
                for p in positions_data if int(p.get("netqty", 0)) != 0
            ]
        except Exception as e:
            return []

    def get_positions(self) -> list[dict]:
        """Alias for get_open_positions for API compatibility."""
        return self.get_open_positions()

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
        strategy_name: str = "Manual",
        symbol_token: str = None
    ) -> dict:
        if not self.is_connected or not self.smart_api:
            return {"status": "REJECTED", "message": "Angel One is not connected. Please provide API Key, Client ID & PIN."}
        try:
            sym = symbol.replace(".NS", "").replace(".BO", "")
            is_nfo = any(k in sym for k in ["CE", "PE", "FUT"])
            exchange = "NFO" if is_nfo else "NSE"
            prod_type = "INTRADAY" if product in ["MIS", "INTRADAY"] else ("CARRYFORWARD" if product == "NRML" else "DELIVERY")
            
            token = symbol_token or "3045" # Defaults to token if provided, else standard symbol
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": sym,
                "symboltoken": str(token),
                "transactiontype": side.upper(),
                "exchange": exchange,
                "ordertype": "MARKET" if order_type == "MARKET" else "LIMIT",
                "producttype": prod_type,
                "duration": "DAY",
                "price": str(price) if price and price > 0 else "0",
                "quantity": str(quantity)
            }
            if sl and sl > 0:
                order_params["stoploss"] = str(round(sl, 2))
                
            order_res = self.smart_api.placeOrder(order_params)
            order_id = order_res.get("data", {}).get("orderid", str(order_res)) if isinstance(order_res, dict) else str(order_res)
            return {
                "status": "FILLED",
                "order_id": order_id,
                "symbol": sym,
                "side": side,
                "quantity": quantity,
                "sl": sl,
                "tp": tp
            }
        except Exception as e:
            return {"status": "FAILED", "message": str(e)}

    def square_off_position(self, symbol: str, reason: str = "MANUAL") -> dict:
        positions = self.get_open_positions()
        sym = symbol.replace(".NS", "").replace(".BO", "")
        for p in positions:
            if p["symbol"] == sym:
                opp_side = "SELL" if p["side"] == "LONG" else "BUY"
                return self.place_order(sym, opp_side, p["quantity"], order_type="MARKET", product="INTRADAY")
        return {"status": "FAILED", "message": f"No open position found for {sym}"}

    def square_off_all(self, reason: str = "MANUAL") -> list[dict]:
        results = []
        for p in self.get_open_positions():
            res = self.square_off_position(p["symbol"], reason=reason)
            results.append(res)
        return results
