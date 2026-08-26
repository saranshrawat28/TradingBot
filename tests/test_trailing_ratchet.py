import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from src.strategies.indicators import (
    calculate_chandelier_exit,
    calculate_trailing_ratchet_levels,
    calculate_atr
)
from src.engine.trade_manager import SmartTradeManager
from src.engine.software_oco_manager import SoftwareOCOManager
from src.brokers.paper_broker import PaperBroker
from src.utils.storage import clear_all_positions, save_position, get_open_positions

class TestTrailingRatchetEngine(unittest.TestCase):
    def setUp(self):
        clear_all_positions()
        self.broker = PaperBroker(initial_capital=100000.0)

    def tearDown(self):
        clear_all_positions()

    def test_chandelier_exit_math(self):
        """Validates that Chandelier exit computes correct high-watermark minus ATR trail."""
        np.random.seed(42)
        n = 30
        close = pd.Series(100.0 + np.cumsum(np.random.normal(0.5, 1.0, n)))
        high = close + np.random.uniform(0.5, 2.0, n)
        low = close - np.random.uniform(0.5, 2.0, n)

        long_trail, short_trail = calculate_chandelier_exit(high, low, close, period=14, multiplier=1.8)
        self.assertEqual(len(long_trail), n)
        self.assertLess(long_trail.iloc[-1], high.iloc[-1])
        self.assertGreater(short_trail.iloc[-1], low.iloc[-1])

    def test_ratchet_stage_transitions(self):
        """Tests the 4-stage ratchet progression: Initial -> Breakeven -> Runner -> Parabolic."""
        entry = 100.0
        initial_risk_r = 2.0 # 1R = Rs 2.0 (SL = 98.0)
        atr_val = 1.5

        # Stage 1: Initial (< +1.0R gain, e.g. price at 101.0)
        r1 = calculate_trailing_ratchet_levels(
            entry_price=entry,
            highest_price=101.0,
            current_price=101.0,
            atr_val=atr_val,
            initial_risk_r=initial_risk_r,
            side="BUY",
            target_1_hit=False
        )
        self.assertEqual(r1["stage"], "INITIAL")
        self.assertEqual(r1["new_sl"], 98.0)

        # Stage 2: Breakeven (+1.0R gain, e.g. price at 102.2)
        r2 = calculate_trailing_ratchet_levels(
            entry_price=entry,
            highest_price=102.2,
            current_price=102.2,
            atr_val=atr_val,
            initial_risk_r=initial_risk_r,
            side="BUY",
            target_1_hit=False
        )
        self.assertEqual(r2["stage"], "BREAKEVEN_LOCKED")
        self.assertGreaterEqual(r2["new_sl"], 100.20) # Breakeven + buffer

        # Stage 3: Target 1 Hit (+1.5R, price at 103.5) -> Runner Chandelier Trailing
        r3 = calculate_trailing_ratchet_levels(
            entry_price=entry,
            highest_price=103.5,
            current_price=103.5,
            atr_val=atr_val,
            initial_risk_r=initial_risk_r,
            side="BUY",
            target_1_hit=True
        )
        self.assertEqual(r3["stage"], "T1_BOOKED_RUNNER_TRAILING")
        self.assertGreaterEqual(r3["new_sl"], 101.50) # Locked minimum floor +0.75R
        self.assertGreaterEqual(r3["locked_r"], 0.75)

        # Stage 4: Parabolic Rider (> +3.0R gain, e.g. peak at 108.0)
        r4 = calculate_trailing_ratchet_levels(
            entry_price=entry,
            highest_price=108.0,
            current_price=107.5,
            atr_val=atr_val,
            initial_risk_r=initial_risk_r,
            side="BUY",
            target_1_hit=True
        )
        self.assertEqual(r4["stage"], "PARABOLIC_RIDER")
        # Peak 108.0 - (1.0 * 1.5 ATR) = 106.5
        self.assertGreaterEqual(r4["new_sl"], 106.5)
        self.assertGreaterEqual(r4["locked_r"], 3.0)

    def test_ratchet_monotonicity_rule(self):
        """Guarantees that trailing stop-loss NEVER drops backwards on price pullbacks."""
        entry = 100.0
        initial_risk_r = 2.0
        atr_val = 1.5

        # Peak reached 106.0
        r_peak = calculate_trailing_ratchet_levels(
            entry_price=entry,
            highest_price=106.0,
            current_price=106.0,
            atr_val=atr_val,
            initial_risk_r=initial_risk_r,
            side="BUY",
            target_1_hit=True,
            current_trailing_sl=0.0
        )
        established_sl = r_peak["new_sl"]

        # Price pulls back to 104.5 (Peak remains 106.0)
        r_pullback = calculate_trailing_ratchet_levels(
            entry_price=entry,
            highest_price=106.0,
            current_price=104.5,
            atr_val=atr_val,
            initial_risk_r=initial_risk_r,
            side="BUY",
            target_1_hit=True,
            current_trailing_sl=established_sl
        )
        self.assertGreaterEqual(r_pullback["new_sl"], established_sl)

    @patch("src.engine.trade_manager.get_live_quote")
    def test_smart_trade_manager_runner_lifecycle(self, mock_quote):
        """Simulates full position lifecycle from entry -> T1 partial profit -> runner trail -> secure exit."""
        # 1. Open mock position in storage
        pos = {
            "symbol": "RELIANCE.NS",
            "quantity": 10,
            "entry_price": 2500.0,
            "current_price": 2500.0,
            "highest_price": 2500.0,
            "sl": 2470.0,
            "target_1": 2545.0, # +1.5R (Risk R = 30)
            "target_2": 2650.0, # +5.0R (Runner Target)
            "side": "BUY",
            "atr": 15.0,
            "initial_risk_r": 30.0,
            "target_1_hit": 0,
            "breakeven_locked": 0,
            "stage": "INITIAL",
            "sl_order_id": "MOCK_SL_123",
            "entry_time": "2026-08-22 09:30:00"
        }
        save_position(pos)

        # Step A: Price reaches +1.0R (Rs 2532)
        mock_quote.return_value = {"price": 2532.0, "symbol": "RELIANCE.NS"}
        events = SmartTradeManager.evaluate_and_manage_positions(self.broker)
        active_pos = get_open_positions()[0]
        self.assertEqual(active_pos["stage"], "BREAKEVEN_LOCKED")
        self.assertGreaterEqual(active_pos["sl"], 2500.0)

        # Step B: Price reaches Target 1 (Rs 2546) -> Book 50% partial profit
        mock_quote.return_value = {"price": 2546.0, "symbol": "RELIANCE.NS"}
        events = SmartTradeManager.evaluate_and_manage_positions(self.broker)
        active_pos = get_open_positions()[0]
        self.assertEqual(active_pos["target_1_hit"], 1)
        self.assertEqual(active_pos["quantity"], 5) # 50% booked (10 -> 5)
        self.assertEqual(active_pos["stage"], "T1_BOOKED_RUNNER_TRAILING")
        self.assertGreaterEqual(active_pos["locked_r"], 0.75)

        # Step C: Price surges to +3.5R (Rs 2610) -> Parabolic rider
        mock_quote.return_value = {"price": 2610.0, "symbol": "RELIANCE.NS"}
        events = SmartTradeManager.evaluate_and_manage_positions(self.broker)
        active_pos = get_open_positions()[0]
        self.assertEqual(active_pos["stage"], "PARABOLIC_RIDER")
        # 2610 - 15 ATR = 2595.0
        self.assertGreaterEqual(active_pos["trailing_sl"], 2595.0)

        # Step D: Price pulls back to Rs 2590 (Hits trailing SL of 2595) -> Secure profit exit
        mock_quote.return_value = {"price": 2590.0, "symbol": "RELIANCE.NS"}
        events = SmartTradeManager.evaluate_and_manage_positions(self.broker)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "TRAILING_SL_EXIT")
        self.assertGreater(events[0]["realized_gain_pct"], 3.0)

    def test_software_oco_modify_sl(self):
        """Tests that SoftwareOCOManager safely handles exchange SL modifications."""
        res = SoftwareOCOManager.modify_exchange_sl_order(
            broker=self.broker,
            symbol="INFY.NS",
            new_sl_price=1850.0,
            sl_order_id="ORD_999"
        )
        self.assertIn(res["status"], ["SUCCESS", "MOCKED_OR_NOT_SUPPORTED"])

if __name__ == "__main__":
    unittest.main()
