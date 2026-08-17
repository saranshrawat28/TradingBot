"""
Brokers factory and exports.
"""

import config
from src.brokers.base_broker import BaseBroker
from src.brokers.paper_broker import PaperBroker
from src.brokers.zerodha_broker import ZerodhaBroker
from src.brokers.zerodha_live import ZerodhaLiveBroker
from src.brokers.angel_broker import AngelOneBroker
from src.brokers.dhan_broker import DhanBroker

BROKERS_MAP = {
    "paper": PaperBroker,
    "zerodha": ZerodhaBroker,
    "zerodha_live": ZerodhaLiveBroker,
    "angel": AngelOneBroker,
    "dhan": DhanBroker
}

def get_broker(broker_name: str = None) -> BaseBroker:
    """Get initialized broker instance."""
    name = (broker_name or config.ACTIVE_BROKER).lower()
    broker_cls = BROKERS_MAP.get(name, PaperBroker)
    return broker_cls()
