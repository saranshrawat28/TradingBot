"""
Zero-Risk Paper Trading Broker with realistic Indian regulatory taxes and fee modeling.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import config
from src.brokers.base_broker import BaseBroker
from src.utils.storage import (
    get_portfolio_state, update_portfolio_state, get_open_positions,
    save_position, delete_position, log_order, log_closed_trade, get_closed_trades
)
from src.data.data_fetcher import get_live_quote
from src.utils.helpers import get_ist_now, clean_symbol

class PaperBroker(BaseBroker):
    """
    Simulated Indian Broker implementing SEBI / NSE fee structure:
    - Brokerage: min(₹20, 0.03%)
    - STT / CTT: 0.025% on Intraday Sell, 0.1% on Delivery
    - Exchange Txn Charges: 0.00345%
    - SEBI Turnover: 0.0001%
    - Stamp Duty: 0.003% on Buy
    - GST: 18% on (Brokerage + Txn Charges + SEBI)
    - Slippage: 0.05%
    """
    
    def __init__(self, initial_capital: float = None):
        super().__init__(name="Paper Trading (Zero Risk)")
        self.is_connected = True
        if initial_capital is not None:
            update_portfolio_state(cash=initial_capital, realized_pnl=0.0)

    def connect(self) -> bool:
        self.is_connected = True
        return True

    def calculate_taxes_and_charges(self, side: str, price: float, quantity: int, product: str = "MIS") -> dict:
        """Calculate complete breakdown of Indian taxes and brokerage for an order."""
        turnover = price * quantity
        fees = config.INDIAN_FEES
        
        # 1. Brokerage
        raw_brokerage = turnover * fees["brokerage_pct"]
        brokerage = min(fees["brokerage_per_order"], raw_brokerage) if product == "MIS" else 0.0 # Zero brokerage on delivery for discount brokers
        
        # 2. STT
        if side.upper() == "SELL":
            stt = turnover * (fees["stt_intraday_sell_pct"] if product == "MIS" else fees["stt_delivery_pct"])
        else:
            stt = turnover * fees["stt_delivery_pct"] if product == "CNC" else 0.0
            
        # 3. Exchange Charges
        exchange_charges = turnover * fees["exchange_txn_charge_pct"]
        
        # 4. SEBI Charges
        sebi_charges = turnover * fees["sebi_turnover_pct"]
        
        # 5. Stamp Duty (on buy only)
        stamp_duty = turnover * fees["stamp_duty_buy_pct"] if side.upper() == "BUY" else 0.0
        
        # 6. GST (18% on Brokerage + Exchange + SEBI)
        gst = (brokerage + exchange_charges + sebi_charges) * fees["gst_pct"]
        
        total_charges = brokerage + stt + exchange_charges + sebi_charges + stamp_duty + gst
        
        return {
            "turnover": round(turnover, 2),
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "exchange_charges": round(exchange_charges, 2),
            "sebi_charges": round(sebi_charges, 2),
            "stamp_duty": round(stamp_duty, 2),
            "gst": round(gst, 2),
            "total_charges": round(total_charges, 2)
        }

    def get_account_balance(self) -> dict:
        state = get_portfolio_state()
        cash = float(state.get("cash", config.DEFAULT_INITIAL_CAPITAL))
        positions = self.get_open_positions()
        
        margin_used = sum(p["entry_price"] * p["quantity"] for p in positions)
        unrealized_pnl = sum(p.get("unrealized_pnl", 0.0) for p in positions)
        total_equity = cash + margin_used + unrealized_pnl
        
        return {
            "cash": round(cash, 2),
            "margin_used": round(margin_used, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(float(state.get("realized_pnl", 0.0)), 2),
            "total_equity": round(total_equity, 2),
            "initial_capital": round(float(state.get("initial_capital", config.DEFAULT_INITIAL_CAPITAL)), 2),
            "open_positions_count": len(positions)
        }

    def get_open_positions(self) -> list[dict]:
        raw_positions = get_open_positions()
        updated_positions = []
        
        for pos in raw_positions:
            sym = pos["symbol"]
            quote = get_live_quote(sym)
            live_p = float(quote.get("price", 0.0))
            current_price = live_p if live_p > 0 else float(pos.get("entry_price", 100.0))
            
            entry_price = float(pos["entry_price"])
            qty = int(pos["quantity"])
            side = pos["side"]
            
            if side.upper() == "BUY" or side.upper() == "LONG":
                pnl = (current_price - entry_price) * qty
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                highest = max(pos.get("highest_price", entry_price), current_price)
            else:
                pnl = (entry_price - current_price) * qty
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                highest = pos.get("highest_price", entry_price)
                
            pos["current_price"] = current_price
            pos["highest_price"] = highest
            pos["unrealized_pnl"] = round(pnl, 2)
            pos["unrealized_pnl_pct"] = round(pnl_pct, 2)
            
            save_position(pos)
            updated_positions.append(pos)
            
        return updated_positions

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
        sym = clean_symbol(symbol)
        
        if quantity <= 0:
            return {"status": "REJECTED", "message": "Quantity must be greater than 0"}
            
        # Get live price if market order
        if price is None or price <= 0:
            quote = get_live_quote(sym)
            market_price = float(quote.get("price", 100.0))
        else:
            market_price = price
            
        # Apply slight slippage
        slippage = market_price * config.INDIAN_FEES["slippage_pct"]
        exec_price = market_price + slippage if side.upper() == "BUY" else market_price - slippage
        exec_price = round(exec_price, 2)
        
        order_value = exec_price * quantity
        tax_info = self.calculate_taxes_and_charges(side, exec_price, quantity, product)
        fee = tax_info["total_charges"]
        
        # Check cash balance
        state = get_portfolio_state()
        cash = float(state.get("cash", config.DEFAULT_INITIAL_CAPITAL))
        
        if side.upper() == "BUY":
            total_required = order_value + fee
            if cash < total_required:
                return {
                    "status": "REJECTED",
                    "message": f"Insufficient funds: Required ₹{total_required:.2f}, Available ₹{cash:.2f}"
                }
            # Deduct cash
            new_cash = cash - total_required
            update_portfolio_state(cash=new_cash, realized_pnl=float(state.get("realized_pnl", 0.0)))
            
            # Save Position
            pos_dict = {
                "symbol": sym,
                "side": "LONG",
                "entry_time": get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST"),
                "entry_price": exec_price,
                "quantity": quantity,
                "current_price": exec_price,
                "sl": sl,
                "tp": tp,
                "trailing_sl": sl,
                "highest_price": exec_price,
                "strategy": strategy_name,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0
            }
            save_position(pos_dict)
            
        elif side.upper() == "SELL":
            # Check if squaring off existing long position
            existing = [p for p in get_open_positions() if p["symbol"] == sym]
            if existing:
                return self.square_off_position(sym, reason=f"Strategy Signal: {strategy_name}")
            else:
                # Direct short sale
                total_required = order_value * 0.2 + fee # 20% intraday margin
                if cash < total_required:
                    return {"status": "REJECTED", "message": f"Insufficient margin for short trade"}
                pos_dict = {
                    "symbol": sym,
                    "side": "SHORT",
                    "entry_time": get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST"),
                    "entry_price": exec_price,
                    "quantity": quantity,
                    "current_price": exec_price,
                    "sl": sl,
                    "tp": tp,
                    "trailing_sl": sl,
                    "highest_price": exec_price,
                    "strategy": strategy_name,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0
                }
                save_position(pos_dict)
                
        # Log Order
        order_dict = {
            "timestamp": get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "symbol": sym,
            "side": side.upper(),
            "order_type": order_type,
            "price": exec_price,
            "quantity": quantity,
            "value": order_value,
            "fee": fee,
            "status": "FILLED",
            "sl": sl,
            "tp": tp,
            "strategy": strategy_name,
            "broker": "paper",
            "notes": f"Taxes: ₹{fee:.2f} | Execution: ₹{exec_price:.2f}"
        }
        order_id = log_order(order_dict)
        
        return {
            "status": "FILLED",
            "order_id": order_id,
            "symbol": sym,
            "side": side,
            "price": exec_price,
            "quantity": quantity,
            "fee": fee,
            "tax_breakdown": tax_info
        }

    def square_off_position(self, symbol: str, reason: str = "MANUAL") -> dict:
        sym = clean_symbol(symbol)
        positions = [p for p in get_open_positions() if p["symbol"] == sym]
        if not positions:
            return {"status": "FAILED", "message": f"No open position found for {sym}"}
            
        pos = positions[0]
        quote = get_live_quote(sym)
        exit_price = float(quote.get("price", pos["entry_price"]))
        
        entry_price = float(pos["entry_price"])
        qty = int(pos["quantity"])
        side = pos["side"]
        
        # Calculate gross and net PnL
        if side == "LONG":
            gross_pnl = (exit_price - entry_price) * qty
            exit_side = "SELL"
        else:
            gross_pnl = (entry_price - exit_price) * qty
            exit_side = "BUY"
            
        tax_info = self.calculate_taxes_and_charges(exit_side, exit_price, qty)
        exit_fee = tax_info["total_charges"]
        net_pnl = gross_pnl - exit_fee
        pnl_pct = (net_pnl / (entry_price * qty)) * 100
        
        # Update Portfolio State
        state = get_portfolio_state()
        cash = float(state.get("cash", config.DEFAULT_INITIAL_CAPITAL))
        current_realized = float(state.get("realized_pnl", 0.0))
        
        new_cash = cash + (entry_price * qty) + net_pnl if side == "LONG" else cash + net_pnl
        new_realized = current_realized + net_pnl
        update_portfolio_state(cash=new_cash, realized_pnl=new_realized)
        
        # Log closed trade
        trade_record = {
            "symbol": sym,
            "side": side,
            "entry_time": pos["entry_time"],
            "exit_time": get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": qty,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "exit_reason": reason,
            "strategy": pos.get("strategy", "Manual"),
            "broker": "paper"
        }
        log_closed_trade(trade_record)
        
        # Log exit order
        log_order({
            "timestamp": trade_record["exit_time"],
            "symbol": sym,
            "side": exit_side,
            "order_type": "MARKET",
            "price": exit_price,
            "quantity": qty,
            "value": exit_price * qty,
            "fee": exit_fee,
            "status": "FILLED",
            "strategy": pos.get("strategy", "Manual"),
            "broker": "paper",
            "notes": f"Square-off: {reason} | PnL: ₹{net_pnl:.2f}"
        })
        
        # Delete from active positions
        delete_position(sym)
        
        return {
            "status": "SUCCESS",
            "symbol": sym,
            "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "exit_price": exit_price,
            "reason": reason
        }

    def partial_close_position(self, symbol: str, quantity_to_close: int, exit_price: Optional[float] = None, reason: str = "Partial Profit") -> dict:
        """Partially close an open position (e.g. 50% profit booking at Target 1)."""
        sym = clean_symbol(symbol)
        positions = [p for p in get_open_positions() if p["symbol"] == sym]
        if not positions:
            return {"status": "FAILED", "message": f"No open position found for {sym}"}
            
        pos = positions[0]
        current_qty = int(pos["quantity"])
        close_qty = min(current_qty, max(1, quantity_to_close))
        
        if close_qty >= current_qty:
            return self.square_off_position(sym, reason=reason)
            
        if exit_price is None or exit_price <= 0:
            quote = get_live_quote(sym)
            exit_price = float(quote.get("price", pos["entry_price"]))
            
        entry_price = float(pos["entry_price"])
        side = pos["side"]
        
        if side == "LONG":
            gross_pnl = (exit_price - entry_price) * close_qty
            exit_side = "SELL"
        else:
            gross_pnl = (entry_price - exit_price) * close_qty
            exit_side = "BUY"
            
        tax_info = self.calculate_taxes_and_charges(exit_side, exit_price, close_qty)
        exit_fee = tax_info["total_charges"]
        net_pnl = gross_pnl - exit_fee
        pnl_pct = (net_pnl / (entry_price * close_qty)) * 100
        
        # Update Portfolio State
        state = get_portfolio_state()
        cash = float(state.get("cash", config.DEFAULT_INITIAL_CAPITAL))
        current_realized = float(state.get("realized_pnl", 0.0))
        
        new_cash = cash + (entry_price * close_qty) + net_pnl if side == "LONG" else cash + net_pnl
        new_realized = current_realized + net_pnl
        update_portfolio_state(cash=new_cash, realized_pnl=new_realized)
        
        # Log partial closed trade
        trade_record = {
            "symbol": sym,
            "side": side,
            "entry_time": pos["entry_time"],
            "exit_time": get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": close_qty,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "exit_reason": f"{reason} ({close_qty}/{current_qty} shares)",
            "strategy": pos.get("strategy", "Manual"),
            "broker": "paper"
        }
        log_closed_trade(trade_record)
        
        # Update remaining position in storage
        pos["quantity"] = current_qty - close_qty
        pos["target_1_hit"] = 1
        pos["stage"] = "BREAKEVEN_LOCKED"
        pos["trailing_sl"] = entry_price # Lock Breakeven
        pos["sl"] = entry_price # Hard SL shifted to entry
        save_position(pos)
        
        return {
            "status": "SUCCESS",
            "symbol": sym,
            "closed_quantity": close_qty,
            "remaining_quantity": pos["quantity"],
            "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "exit_price": exit_price,
            "reason": reason
        }

    def get_positions(self) -> list[dict]:
        """Fetch open positions in standard broker format."""
        return get_open_positions()

    def get_margins(self) -> dict:
        """Fetch cash and margin in standard broker format."""
        state = get_portfolio_state()
        cash = float(state.get("cash", config.DEFAULT_INITIAL_CAPITAL))
        return {"available_cash": cash, "used_margin": 0.0}

    @property
    def capital(self) -> float:
        state = get_portfolio_state()
        return float(state.get("cash", config.DEFAULT_INITIAL_CAPITAL))

    @property
    def closed_trades(self) -> list[dict]:
        return get_closed_trades()

    def square_off_all(self, reason: str = "MANUAL") -> list[dict]:
        results = []
        for pos in get_open_positions():
            res = self.square_off_position(pos["symbol"], reason=reason)
            results.append(res)
        return results
