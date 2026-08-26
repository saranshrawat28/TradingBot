"""
Risk Management Engine for Indian Equities and F&O Trading.
Protects capital with dynamic position sizing, ATR stop loss, trailing stops, and circuit breakers.
"""

from datetime import datetime
import config
from src.utils.helpers import is_intraday_squareoff_time, get_ist_now
from src.engine.ai_guardrails import AIGuardrails

class RiskManager:
    """
    Institutional risk controls:
    - Max risk per trade (% of portfolio equity)
    - ATR-based volatility stop-loss & target calculations
    - Trailing stop-loss update logic
    - Daily loss circuit breaker
    - 3:15 PM IST intraday auto square-off
    """
    
    def __init__(
        self,
        risk_per_trade_pct: float = config.DEFAULT_RISK_PER_TRADE_PCT,
        max_daily_loss_pct: float = config.MAX_DAILY_LOSS_PCT,
        default_sl_pct: float = config.DEFAULT_STOP_LOSS_PCT,
        default_tp_pct: float = config.DEFAULT_TAKE_PROFIT_PCT,
        trailing_sl_pct: float = config.DEFAULT_TRAILING_SL_PCT
    ):
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.default_sl_pct = default_sl_pct
        self.default_tp_pct = default_tp_pct
        self.trailing_sl_pct = trailing_sl_pct

    def calculate_position_size(
        self,
        total_equity: float,
        entry_price: float,
        stop_loss_price: float = None,
        max_capital_allocation_pct: float = 20.0
    ) -> int:
        """
        Calculate maximum safe share quantity to purchase.
        Ensures max capital risk is capped at `risk_per_trade_pct`.
        """
        if total_equity <= 0 or entry_price <= 0:
            return 0
            
        sl_p = stop_loss_price if (stop_loss_price and stop_loss_price > 0) else entry_price * (1.0 - (self.default_sl_pct / 100.0))
        return AIGuardrails.calculate_dynamic_position_size(
            capital=total_equity,
            entry_price=entry_price,
            sl_price=sl_p,
            lot_size=1,
            max_lots_cap=999999,
            risk_pct=self.risk_per_trade_pct / 100.0,
            max_capital_cap=max_capital_allocation_pct / 100.0
        )

    def calculate_sl_tp_prices(
        self,
        side: str,
        entry_price: float,
        atr_value: float = None,
        atr_multiplier_sl: float = 1.5,
        risk_reward_ratio: float = 2.0
    ) -> tuple[float, float]:
        """
        Calculate dynamic Stop Loss (SL) and Take Profit (TP) levels.
        Uses ATR volatility if available, otherwise default percentage.
        """
        return AIGuardrails.calculate_sl_tp_prices(
            side=side,
            entry_price=entry_price,
            atr_value=atr_value,
            atr_multiplier_sl=atr_multiplier_sl,
            risk_reward_ratio=risk_reward_ratio,
            default_sl_pct=self.default_sl_pct,
            default_tp_pct=self.default_tp_pct
        )

    def update_trailing_stop(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        highest_price: float,
        current_sl: float
    ) -> float:
        """
        Calculate updated trailing stop-loss price as profit grows.
        """
        return AIGuardrails.update_trailing_stop(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            highest_price=highest_price,
            current_sl=current_sl,
            trailing_sl_pct=self.trailing_sl_pct
        )

    def is_daily_circuit_breaker_triggered(self, initial_capital: float, current_equity: float) -> tuple[bool, str]:
        """Check if daily drawdown exceeded maximum allowed risk limit."""
        if initial_capital <= 0:
            return False, "OK"
            
        drawdown_pct = ((initial_capital - current_equity) / initial_capital) * 100.0
        if drawdown_pct >= self.max_daily_loss_pct:
            return True, f"CIRCUIT BREAKER HIT: Daily drawdown {drawdown_pct:.2f}% reached limit ({self.max_daily_loss_pct}%). Trading halted."
            
        return False, f"Drawdown {drawdown_pct:.2f}% within safe limit ({self.max_daily_loss_pct}%)"

    def should_square_off_intraday(self) -> bool:
        """Check if intraday auto square-off time (15:15 IST) is reached."""
        return is_intraday_squareoff_time()
