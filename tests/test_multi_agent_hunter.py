"""
Unit & Verification Tests for Multi-Agent AI Strategy Council, Software OCO Manager, and Market Hunter Daemon.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from src.ai.multi_agent_council import MultiAgentCouncil
from src.engine.software_oco_manager import SoftwareOCOManager
from src.engine.market_hunter_daemon import MarketHunterDaemon
from src.brokers.paper_broker import PaperBroker

class TestMultiAgentHunterSuite(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 100-candle 5m DataFrame
        np.random.seed(42)
        n = 100
        close = 1000.0 + np.cumsum(np.random.randn(n) * 2.0)
        high = close + np.random.rand(n) * 5.0
        low = close - np.random.rand(n) * 5.0
        open_p = close + np.random.randn(n)
        vol = np.random.randint(10000, 50000, n)

        self.df = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": vol
        })
        self.quote = {
            "price": float(close[-1]),
            "previous_close": float(close[-2]),
            "bid": float(close[-1] * 0.999),
            "ask": float(close[-1] * 1.001)
        }

    def test_multi_agent_council_stage_1_prefilter_blocks_weak_math(self):
        """
        Stage 1 Pre-Filter: Math Score < 7.0 is blocked immediately (0 LLM cost).
        """
        with patch("src.engine.stock_advisor.StockAdvisor.evaluate_df_slice", return_value={"score": 5.5, "verdict": "NEUTRAL"}):
            res = MultiAgentCouncil.evaluate_candidate("RELIANCE.NS", self.df, self.quote)
            self.assertFalse(res["passed_prefilter"])
            self.assertFalse(res["consensus_approved"])
            self.assertIn("below Stage-1 pre-filter", res["rejection_reason"])

    def test_multi_agent_council_asymmetric_defense_veto(self):
        """
        Asymmetric Defense Veto: Agent 2 Veto (e.g. illiquidity spread trap) forces consensus to 0.0.
        """
        illiquid_quote = dict(self.quote)
        illiquid_quote["bid"] = 1000.0
        illiquid_quote["ask"] = 1020.0 # 2.0% spread trap (> 1.2% threshold)

        with patch("src.engine.stock_advisor.StockAdvisor.evaluate_df_slice", return_value={"score": 8.5, "verdict": "STRONG_BUY"}):
            res = MultiAgentCouncil.evaluate_candidate("RELIANCE.NS", self.df, illiquid_quote)
            self.assertTrue(res["passed_prefilter"])
            self.assertFalse(res["consensus_approved"])
            self.assertEqual(res["consensus_score"], 0.0)
            self.assertEqual(res["verdict"], "REJECTED_BY_DEFENSE_VETO")
            self.assertTrue(res["agents"]["agent_2_defense"]["veto"])

    def test_multi_agent_council_consensus_approval(self):
        """
        Consensus Approval: When Math >= 7.50, weighted Council Score >= 7.50, and S2 >= 6.0 -> APPROVED.
        """
        with patch("src.engine.stock_advisor.StockAdvisor.evaluate_df_slice", return_value={"score": 8.2, "verdict": "STRONG_BUY"}):
            res = MultiAgentCouncil.evaluate_candidate("TMCV.NS", self.df, self.quote)
            self.assertTrue(res["passed_prefilter"])
            self.assertTrue(res["consensus_approved"])
            self.assertGreaterEqual(res["consensus_score"], 7.50)
            self.assertEqual(res["verdict"], "APPROVED")

    def test_software_oco_manager_entry_and_sl_m_placement(self):
        """
        Verify SoftwareOCOManager places regular entry + independent SL-M order on broker.
        """
        broker = PaperBroker()
        broker.cash = 100000.0

        res = SoftwareOCOManager.execute_guarded_entry_with_oco(
            broker=broker,
            symbol="RELIANCE.NS",
            side="BUY",
            quantity=10,
            entry_price=2500.0,
            sl_price=2450.0,
            target_1_price=2575.0
        )

        self.assertEqual(res["status"], "FILLED")
        self.assertEqual(res["quantity"], 10)
        self.assertIsNotNone(res["sl_order_id"])
        self.assertEqual(res["sl_price"], 2450.0)

    def test_software_oco_crash_recovery(self):
        """
        Crash Recovery: Detects unhedged open position and auto-places safety SL-M.
        """
        from src.utils.storage import save_position, clear_all_positions
        clear_all_positions()
        
        broker = PaperBroker()
        # Simulate open position created before crash without sl_order_id
        save_position({
            "symbol": "INFY.NS",
            "quantity": 25,
            "entry_time": "2026-08-18 09:30:00 IST",
            "entry_price": 1500.0,
            "current_price": 1510.0,
            "side": "BUY",
            "sl": 1475.0,
            "sl_order_id": None
        })

        recovered = SoftwareOCOManager.check_and_recover_unhedged_positions(broker)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["symbol"], "INFY.NS")
        self.assertEqual(recovered[0]["status"], "HEDGED_AFTER_CRASH")

if __name__ == "__main__":
    unittest.main()
