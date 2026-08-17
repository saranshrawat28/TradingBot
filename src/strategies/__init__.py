"""
Strategies package initialization and registry.
"""

from src.strategies.base_strategy import BaseStrategy
from src.strategies.ema_rsi_strategy import EmaRsiStrategy
from src.strategies.macd_strategy import MacdStrategy
from src.strategies.bollinger_strategy import BollingerBandsStrategy
from src.strategies.supertrend_strategy import SuperTrendStrategy
from src.strategies.multi_indicator import MultiIndicatorConfluenceStrategy

AVAILABLE_STRATEGIES = {
    "EMA Crossover + RSI": EmaRsiStrategy,
    "MACD Momentum": MacdStrategy,
    "Bollinger Bands": BollingerBandsStrategy,
    "SuperTrend (Intraday/Swing)": SuperTrendStrategy,
    "Multi-Indicator Confluence": MultiIndicatorConfluenceStrategy,
}

def get_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """Instantiate a strategy by name with optional parameters."""
    strategy_cls = AVAILABLE_STRATEGIES.get(strategy_name, EmaRsiStrategy)
    return strategy_cls(**kwargs)
