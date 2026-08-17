"""
Automated Test Suite for Smart Trade Lifecycle Manager and Trailing Stop-Loss Engine.
"""

import unittest
from unittest.mock import patch, MagicMock
from src.engine.trade_manager import SmartTradeManager
from src.engine.auto_pilot_daemon import AutoPilotDaemon
from src.brokers.paper_broker import PaperBroker
from src.utils.storage import reset_all_data, get_open_positions, save_position

class TestSmartTradeManager(unittest.TestCase):

    def setUp(self):
        reset_all_data(100000.0)
        self.broker = PaperBroker(initial_capital=100000.0)

    @patch("src.engine.trade_manager.get_live_quote")
    def test_target_1_partial_profit_and_breakeven_lock(self, mock_quote):
        # 1. Place a test buy order: 50 shares of NIFTY @ ₹100
        # Target 1 = ₹120 (+20%), Target 2 = ₹150 (+50%), SL = ₹85 (-15%)
        pos_dict = {
            "symbol": "NIFTY 24500 CE",
            "side": "LONG",
            "entry_time": "2026-08-16 10:00:00 IST",
            "entry_price": 100.0,
            "quantity": 50,
            "current_price": 100.0,
            "sl": 85.0,
            "tp": 120.0,
            "target_1": 120.0,
            "target_2": 150.0,
            "trailing_sl": 85.0,
            "highest_price": 100.0,
            "strategy": "AI_Radar",
            "target_1_hit": 0,
            "stage": "ACTIVE"
        }
        save_position(pos_dict)

        # 2. Simulate price rising to ₹122 (Target 1 Reached)
        mock_quote.return_value = {"price": 122.0, "change_pct": 22.0}

        events = SmartTradeManager.evaluate_and_manage_positions(self.broker)
        
        # Verify 50% profit was booked
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "TARGET_1_PROFIT_BOOKED")
        self.assertEqual(events[0]["closed_qty"], 25)

        # Verify remaining position
        open_pos = get_open_positions()
        self.assertEqual(len(open_pos), 1)
        rem_pos = open_pos[0]
        self.assertEqual(rem_pos["quantity"], 25)
        self.assertEqual(rem_pos["target_1_hit"], 1)
        self.assertEqual(rem_pos["stage"], "BREAKEVEN_LOCKED")
        self.assertEqual(rem_pos["trailing_sl"], 100.0) # Breakeven locked!

    @patch("src.engine.trade_manager.get_live_quote")
    def test_target_2_full_exit(self, mock_quote):
        # Position with Target 1 already hit, 25 shares remaining
        pos_dict = {
            "symbol": "NIFTY 24500 CE",
            "side": "LONG",
            "entry_time": "2026-08-16 10:00:00 IST",
            "entry_price": 100.0,
            "quantity": 25,
            "current_price": 125.0,
            "sl": 100.0,
            "tp": 150.0,
            "target_1": 120.0,
            "target_2": 150.0,
            "trailing_sl": 100.0,
            "highest_price": 125.0,
            "strategy": "AI_Radar",
            "target_1_hit": 1,
            "stage": "BREAKEVEN_LOCKED"
        }
        save_position(pos_dict)

        # Simulate price hitting Target 2 @ ₹152
        mock_quote.return_value = {"price": 152.0, "change_pct": 52.0}

        events = SmartTradeManager.evaluate_and_manage_positions(self.broker)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "TARGET_2_FULL_EXIT")

        # Position should now be completely closed
        self.assertEqual(len(get_open_positions()), 0)

    @patch("src.engine.trade_manager.get_live_quote")
    def test_trailing_stop_loss_exit_above_breakeven(self, mock_quote):
        # Position with Target 1 hit, price went to ₹140, trailing SL at ₹128.80
        pos_dict = {
            "symbol": "NIFTY 24500 CE",
            "side": "LONG",
            "entry_time": "2026-08-16 10:00:00 IST",
            "entry_price": 100.0,
            "quantity": 25,
            "current_price": 138.0,
            "sl": 128.80,
            "tp": 150.0,
            "target_1": 120.0,
            "target_2": 150.0,
            "trailing_sl": 128.80,
            "highest_price": 140.0,
            "strategy": "AI_Radar",
            "target_1_hit": 1,
            "stage": "BREAKEVEN_LOCKED"
        }
        save_position(pos_dict)

        # Price drops back down to ₹125 (violating trailing SL of ₹128.80)
        mock_quote.return_value = {"price": 125.0, "change_pct": 25.0}

        events = SmartTradeManager.evaluate_and_manage_positions(self.broker, trailing_buffer_pct=8.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "TRAILING_SL_EXIT")
        self.assertEqual(len(get_open_positions()), 0)

if __name__ == "__main__":
    unittest.main()
