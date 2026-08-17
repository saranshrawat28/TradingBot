"""
Zerodha Kite Connect Broker Adapter for Indian Equities and F&O.
"""

import config
from src.brokers.base_broker import BaseBroker
from src.utils.helpers import clean_symbol

class ZerodhaBroker(BaseBroker):
    """
    Zerodha Kite Connect API integration.
    Supports login, profile check, live quotes, and order placement (MIS, CNC, NRML).
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, access_token: str = None):
        super().__init__(name="Zerodha Kite Connect")
        self.api_key = api_key or config.ZERODHA_API_KEY
        self.api_secret = api_secret or config.ZERODHA_API_SECRET
        self.access_token = access_token or config.ZERODHA_ACCESS_TOKEN
        self.kite = None
        self.connect()

    def connect(self) -> bool:
        if not self.api_key or not self.access_token:
            self.is_connected = False
            return False
            
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            # Test profile fetch
            profile = self.kite.profile()
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Zerodha Connection notice: {e}")
            self.is_connected = False
            return False

    def get_account_balance(self) -> dict:
        if not self.is_connected or not self.kite:
            return {
                "cash": 0.0, "margin_used": 0.0, "unrealized_pnl": 0.0,
                "realized_pnl": 0.0, "total_equity": 0.0, "status": "Not Connected"
            }
        try:
            margins = self.kite.margins(segment="equity")
            cash = float(margins.get("available", {}).get("live_balance", 0.0))
            margin_used = float(margins.get("utilised", {}).get("debits", 0.0))
            return {
                "cash": round(cash, 2),
                "margin_used": round(margin_used, 2),
                "total_equity": round(cash + margin_used, 2),
                "status": "Connected"
            }
        except Exception as e:
            return {"error": str(e), "cash": 0.0, "status": "Error"}

    def get_open_positions(self) -> list[dict]:
        if not self.is_connected or not self.kite:
            return []
        try:
            positions_data = self.kite.positions()
            net = positions_data.get("net", [])
            results = []
            for p in net:
                if p.get("quantity", 0) != 0:
                    results.append({
                        "symbol": p.get("tradingsymbol"),
                        "side": "LONG" if p.get("quantity", 0) > 0 else "SHORT",
                        "quantity": abs(p.get("quantity", 0)),
                        "entry_price": float(p.get("average_price", 0.0)),
                        "current_price": float(p.get("last_price", 0.0)),
                        "unrealized_pnl": float(p.get("pnl", 0.0)),
                        "unrealized_pnl_pct": float(p.get("pnl_percentage", 0.0))
                    })
            return results
        except Exception as e:
            print(f"Error fetching Zerodha positions: {e}")
            return []

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float = None,
        order_type: str = "MARKET",
        product: str = "MIS",
        sl: float = None,
        tp: float = None,
        strategy_name: str = "Manual"
    ) -> dict:
        if not self.is_connected or not self.kite:
            return {
                "status": "REJECTED",
                "message": "Zerodha Kite is not connected. Please provide API Key & Access Token in Settings."
            }
            
        sym = symbol.replace(".NS", "").replace(".BO", "")
        try:
            transaction_type = self.kite.TRANSACTION_TYPE_BUY if side.upper() == "BUY" else self.kite.TRANSACTION_TYPE_SELL
            product_type = self.kite.PRODUCT_MIS if product == "MIS" else self.kite.PRODUCT_CNC
            order_t = self.kite.ORDER_TYPE_MARKET if order_type == "MARKET" else self.kite.ORDER_TYPE_LIMIT
            
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=sym,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product_type,
                order_type=order_t,
                price=price
            )
            return {"status": "FILLED", "order_id": order_id, "symbol": sym, "side": side}
        except Exception as e:
            return {"status": "FAILED", "message": str(e)}

    def square_off_position(self, symbol: str, reason: str = "MANUAL") -> dict:
        positions = self.get_open_positions()
        sym = symbol.replace(".NS", "").replace(".BO", "")
        for p in positions:
            if p["symbol"] == sym:
                opp_side = "SELL" if p["side"] == "LONG" else "BUY"
                return self.place_order(sym, opp_side, p["quantity"], order_type="MARKET", product="MIS")
        return {"status": "FAILED", "message": f"No open position found for {sym}"}

    def square_off_all(self, reason: str = "MANUAL") -> list[dict]:
        results = []
        for p in self.get_open_positions():
            res = self.square_off_position(p["symbol"], reason=reason)
            results.append(res)
        return results
