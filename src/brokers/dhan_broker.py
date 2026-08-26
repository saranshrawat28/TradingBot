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

    def get_margins(self) -> dict:
        """Fetch available margin for state reconciliation."""
        if not self.is_connected or not self.dhan:
            return {"available_cash": 100000.0, "used_margin": 0.0}
        try:
            limits = self.dhan.get_fund_limits()
            data = limits.get("data", {})
            avail = float(data.get("availabelBalance", 100000.0))
            return {"available_cash": avail, "used_margin": 0.0}
        except Exception as e:
            return {"available_cash": 100000.0, "used_margin": 0.0}

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
                    "unrealized_pnl": float(p.get("realizedProfit", 0.0)),
                    "product": p.get("productType", "INTRADAY")
                }
                for p in data if int(p.get("netQty", 0)) != 0
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
        security_id: str = "1333"
    ) -> dict:
        if not self.is_connected or not self.dhan:
            return {"status": "REJECTED", "message": "DhanHQ is not connected. Please provide Client ID & Access Token."}
        try:
            sym = symbol.replace(".NS", "").replace(".BO", "")
            is_nfo = any(k in sym for k in ["CE", "PE", "FUT"])
            exch_segment = "NSE_FNO" if is_nfo else "NSE_EQ"
            prod_type = "INTRADAY" if product in ["MIS", "INTRADAY"] else ("MARGIN" if product == "NRML" else "CNC")
            
            order_data = self.dhan.place_order(
                security_id=security_id,
                exchange_segment=exch_segment,
                transaction_type=self.dhan.BUY if side.upper() == "BUY" else self.dhan.SELL,
                quantity=quantity,
                order_type=self.dhan.MARKET if order_type == "MARKET" else self.dhan.LIMIT,
                product_type=prod_type,
                price=price if price and price > 0 else 0,
                trigger_price=sl if sl and sl > 0 else 0
            )
            order_id = order_data.get("data", {}).get("orderId", str(order_data)) if isinstance(order_data, dict) else str(order_data)
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
