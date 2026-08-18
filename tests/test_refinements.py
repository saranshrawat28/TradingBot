"""
Unit Test Suite for ApexTrade Refinements & Symmetries:
1. Classical & Fibonacci Pivot Point Arithmetic.
2. Symmetric 4-Case Pivot Confluence Evaluator.
3. Fully-Specified 4-Zone VWAP Location Evaluator.
4. Trailing High-Water Mark (HWM) Daily Profit Lock & Breakeven Direction.
5. Stagnant Trade Chop Timeout in SmartTradeManager.
6. Analytical Options Expiration Payoff Curves.
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pandas as pd
import numpy as np

from src.strategies.indicators import (
    calculate_classical_pivots, calculate_fibonacci_pivots,
    evaluate_vwap_location_score, evaluate_pivot_confluence
)
from src.engine.ai_guardrails import AIGuardrails
from src.engine.trade_manager import SmartTradeManager
from src.brokers.paper_broker import PaperBroker
from src.strategies.options_greeks import SmartStrikeSelector
from src.utils.storage import reset_all_data, save_position

class TestRefinementsSuite(unittest.TestCase):

    def setUp(self):
        reset_all_data(100000.0)

    def test_pivot_points_arithmetic(self):
        """Verify classical and Fibonacci pivot calculations against textbook reference."""
        # Previous Day: High = 110, Low = 90, Close = 100 (Range = 20)
        # P = (110 + 90 + 100)/3 = 100.0
        # Classical: R1 = 200 - 90 = 110, S1 = 200 - 110 = 90, R2 = 100 + 20 = 120, S2 = 100 - 20 = 80
        piv = calculate_classical_pivots(110.0, 90.0, 100.0)
        self.assertEqual(piv["pivot"], 100.0)
        self.assertEqual(piv["r1"], 110.0)
        self.assertEqual(piv["s1"], 90.0)
        self.assertEqual(piv["r2"], 120.0)
        self.assertEqual(piv["s2"], 80.0)

        # Fibonacci: R1 = 100 + 0.382*20 = 107.64, S1 = 100 - 0.382*20 = 92.36
        fib = calculate_fibonacci_pivots(110.0, 90.0, 100.0)
        self.assertEqual(fib["pivot"], 100.0)
        self.assertEqual(fib["fib_r1"], 107.64)
        self.assertEqual(fib["fib_s1"], 92.36)

    def test_symmetric_4_case_pivot_confluence(self):
        """Verify exact symmetric 4-case scoring in evaluate_pivot_confluence."""
        pivots = {"pivot": 100.0, "r1": 105.0, "s1": 95.0, "r2": 110.0, "s2": 90.0}

        # 1. Bullish + Near Support (95.10 vs S1=95.0, 0.1% dist) -> +0.30
        c1 = evaluate_pivot_confluence(95.10, pivots, raw_trend=1.5)
        self.assertEqual(c1, 0.30)

        # 2. Bullish + Near Resistance (104.90 vs R1=105.0, 0.1% dist) -> -0.30
        c2 = evaluate_pivot_confluence(104.90, pivots, raw_trend=1.5)
        self.assertEqual(c2, -0.30)

        # 3. Bearish + Near Resistance (104.90 vs R1=105.0) -> -0.30 (confirms short thesis)
        c3 = evaluate_pivot_confluence(104.90, pivots, raw_trend=-1.5)
        self.assertEqual(c3, -0.30)

        # 4. Bearish + Near Support (95.10 vs S1=95.0) -> +0.30 (demand obstacle vs short)
        c4 = evaluate_pivot_confluence(95.10, pivots, raw_trend=-1.5)
        self.assertEqual(c4, 0.30)

        # 5. Far from any level (100.0 vs S1=95/R1=105, 5% dist) -> 0.00
        c5 = evaluate_pivot_confluence(100.0, pivots, raw_trend=1.5)
        self.assertEqual(c5, 0.00)

    def test_4_zone_vwap_location_scoring(self):
        """Verify 4-Zone VWAP location logic: ungated mean reversion, gated value, and exhaustion."""
        vwap_bands = {
            "vwap": 100.0,
            "std": 2.0,
            "upper_1sigma": 102.0,
            "lower_1sigma": 98.0,
            "upper_2sigma": 104.0,
            "lower_2sigma": 96.0
        }

        # 1. Mean-reversion (+-0.5 sigma, 99.0 to 101.0) -> +0.80 across Bullish, Bearish, and Neutral!
        self.assertEqual(evaluate_vwap_location_score(100.2, vwap_bands, raw_trend=1.0), 0.80)
        self.assertEqual(evaluate_vwap_location_score(100.2, vwap_bands, raw_trend=-1.0), 0.80)
        self.assertEqual(evaluate_vwap_location_score(100.2, vwap_bands, raw_trend=0.0), 0.80)

        # 2. Discount Support Zone (98.0, which is in [VWAP - 1.5s, VWAP - 0.5s] = [97.0, 99.0])
        # Bullish setup gets discount bonus (+1.20)
        self.assertEqual(evaluate_vwap_location_score(98.0, vwap_bands, raw_trend=1.5), 1.20)
        # Bearish setup gets NO discount bonus (it is breaking down, so score is -0.20, NOT +1.20!)
        self.assertNotEqual(evaluate_vwap_location_score(98.0, vwap_bands, raw_trend=-1.5), 1.20)

        # 3. Premium Resistance Zone (102.0, which is in [VWAP + 0.5s, VWAP + 1.5s] = [101.0, 103.0])
        # Bearish setup gets premium short location (-1.20)
        self.assertEqual(evaluate_vwap_location_score(102.0, vwap_bands, raw_trend=-1.5), -1.20)
        # Bullish setup does not get discount bonus (+0.20 mild continuation)
        self.assertEqual(evaluate_vwap_location_score(102.0, vwap_bands, raw_trend=1.5), 0.20)

        # 4. Overextended Exhaustion (> VWAP + 2.0s = 104.0) -> -0.80 penalty
        self.assertEqual(evaluate_vwap_location_score(105.5, vwap_bands, raw_trend=1.5), -0.80)

    def test_hwm_daily_profit_lock_circuit_breaker(self):
        """Verify Trailing High-Water Mark circuit breaker halts new entries when profit drops below 50% floor."""
        guard = AIGuardrails(max_daily_loss_flat=2000.0, min_confidence_threshold=7.5)
        portfolio = {"capital": 100000.0, "initial_capital": 100000.0, "daily_pnl": 0.0, "open_positions": []}
        proposal = {"action": "BUY_STOCK", "target_asset": "RELIANCE", "confidence_score": 8.5}

        # 1. Normal state (PnL 0) -> Approved
        app, _, _ = guard.evaluate_proposal(proposal, portfolio)
        self.assertTrue(app)

        # 2. PnL rises to +₹3,000 (HWM = 3,000, Floor = 1,500) -> Approved
        portfolio["daily_pnl"] = 3000.0
        app, _, _ = guard.evaluate_proposal(proposal, portfolio)
        self.assertTrue(app)
        self.assertEqual(guard.daily_high_water_mark, 3000.0)

        # 3. PnL drops to +₹1,400 (<= 50% Floor of 1,500) -> Circuit Breaker FIRES!
        portfolio["daily_pnl"] = 1400.0
        app, reason, _ = guard.evaluate_proposal(proposal, portfolio)
        self.assertFalse(app)
        self.assertIn("Trailing Daily Profit Lock", reason)
        self.assertTrue(guard.circuit_broken)

    def test_breakeven_sl_calculation_branching(self):
        """Verify Breakeven SL calculation for long asset vs short equity."""
        # Long Stock / Long Call / Long Put: Entry 100 -> SL = 100 * 1.002 = 100.20
        be_long = AIGuardrails.calculate_breakeven_sl(100.0, side="BUY", is_option=False, buffer_pct=0.002)
        self.assertEqual(be_long, 100.20)

        be_opt = AIGuardrails.calculate_breakeven_sl(50.0, side="BUY", is_option=True, buffer_pct=0.002)
        self.assertEqual(be_opt, 50.10)

        # Short Stock: Entry 100 -> SL = 100 * (1 - 0.002) = 99.80
        be_short = AIGuardrails.calculate_breakeven_sl(100.0, side="SELL", is_option=False, buffer_pct=0.002)
        self.assertEqual(be_short, 99.80)

    def test_stagnant_chop_timeout_exit(self):
        """Verify that positions inactive for >=45 mins in chop trigger STAGNANT_CHOP_TIMEOUT."""
        broker = PaperBroker(initial_capital=100000.0)
        
        # Open a position entered 50 minutes ago
        old_time = (datetime.now() - timedelta(minutes=50)).isoformat()
        pos = {
            "symbol": "TCS",
            "quantity": 10,
            "entry_time": old_time,
            "entry_price": 100.0,
            "current_price": 100.5, # Tiny 0.05R progress
            "highest_price": 100.5,
            "sl": 90.0,
            "target_1": 115.0,
            "target_2": 125.0,
            "trailing_sl": 90.0,
            "side": "BUY",
            "target_1_hit": 0,
            "breakeven_locked": 0
        }
        save_position(pos)

        with patch("src.engine.trade_manager.get_live_quote", return_value={"price": 100.5}):
            events = SmartTradeManager.evaluate_and_manage_positions(broker)

        stagnant_events = [e for e in events if e["type"] == "STAGNANT_CHOP_TIMEOUT"]
        self.assertEqual(len(stagnant_events), 1)
        self.assertEqual(stagnant_events[0]["symbol"], "TCS")

    def test_options_payoff_curve_analytics(self):
        """Verify analytical option payoff curves for Calls and Puts."""
        # Buy Call: Spot 100, Strike 100, Premium 5, Qty 10
        # Breakeven = 105, Max loss = -50, At Spot 110: PnL = 10 * (10 - 5) = +50
        call_payoff = SmartStrikeSelector.calculate_payoff_curve(
            spot_price=100.0,
            strike=100.0,
            premium=5.0,
            action="BUY_CALL",
            quantity=10
        )
        self.assertEqual(call_payoff["breakeven"], 105.0)
        self.assertEqual(call_payoff["max_loss"], -50.0)
        self.assertEqual(call_payoff["max_profit"], "Unlimited")

        # Buy Put: Spot 100, Strike 100, Premium 5, Qty 10
        # Breakeven = 95, Max loss = -50, At Spot 90: PnL = 10 * (10 - 5) = +50
        put_payoff = SmartStrikeSelector.calculate_payoff_curve(
            spot_price=100.0,
            strike=100.0,
            premium=5.0,
            action="BUY_PUT",
            quantity=10
        )
        self.assertEqual(put_payoff["breakeven"], 95.0)
        self.assertEqual(put_payoff["max_loss"], -50.0)
        self.assertEqual(put_payoff["max_profit"], 950.0)

if __name__ == "__main__":
    unittest.main()
