"""
Abstract Broker Interface for Indian Stock Brokers (Paper, Zerodha, Angel One, Dhan).
"""

from abc import ABC, abstractmethod

class BaseBroker(ABC):
    """
    Abstract interface for order routing and account synchronization.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection or validate credentials with the broker."""
        pass
        
    @abstractmethod
    def get_account_balance(self) -> dict:
        """
        Return dict with:
        {'cash': float, 'margin_used': float, 'total_equity': float}
        """
        pass
        
    @abstractmethod
    def get_open_positions(self) -> list[dict]:
        """Return list of open positions."""
        pass
        
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str, # "BUY" or "SELL"
        quantity: int,
        price: float = None,
        order_type: str = "MARKET", # "MARKET", "LIMIT"
        product: str = "MIS", # "MIS" (Intraday), "CNC" (Delivery), "NRML" (F&O)
        sl: float = None,
        tp: float = None,
        strategy_name: str = "Manual"
    ) -> dict:
        """Place an order with the broker."""
        pass
        
    @abstractmethod
    def square_off_position(self, symbol: str, reason: str = "MANUAL") -> dict:
        """Square off an existing open position."""
        pass
        
    @abstractmethod
    def square_off_all(self, reason: str = "MANUAL") -> list[dict]:
        """Square off all active open positions."""
        pass
