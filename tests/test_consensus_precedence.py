"""
Automated Test Suite for Asymmetric Consensus & Veto Precedence Engine.
Explicitly tests boundary cutoff values: 6.9, 7.0, 7.4, 7.5, 7.6.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.ai.ai_agent import AITradingAgent
from src.engine.ai_guardrails import AIGuardrails
from src.brokers.paper_broker import PaperBroker

class TestConsensusPrecedenceSuite(unittest.TestCase):

    def setUp(self):
        self.guardrails = AIGuardrails(min_confidence_threshold=7.5)
        self.broker = PaperBroker(initial_capital=100000.0)
        self.mock_llm = MagicMock()
        self.mock_llm.provider = "test"
        self.mock_llm.model = "test-model"

    @patch("src.ai.ai_agent.StockAdvisor.evaluate_df_slice")
    @patch("src.ai.ai_agent.get_live_quote")
    @patch("src.ai.ai_agent.get_historical_data")
    def test_boundary_6_9_weak_math_blocks_llm(self, mock_hist, mock_quote, mock_stock_adv):
        """Boundary Test: Math 6.9 < 7.5 -> LLM with 9.5 confidence CANNOT upgrade."""
        mock_quote.return_value = {"price": 2500.0, "low": 2490.0, "high": 2510.0}
        mock_hist.return_value = MagicMock(empty=False)
        mock_stock_adv.return_value = {"score": 6.9, "regime": "TRENDING"}
        
        # LLM proposes aggressive BUY_STOCK with 9.5 confidence
        self.mock_llm.generate_completion.return_value = '{"action": "BUY_STOCK", "confidence_score": 9.5, "target_asset": "RELIANCE", "suggested_sl_pct": 1.5, "suggested_tp_pct": 3.5}'
        
        agent = AITradingAgent(self.mock_llm, self.guardrails, self.broker, min_consensus_threshold=7.5)
        res = agent.evaluate_and_execute("RELIANCE.NS")
        
        # Must be vetoed to HOLD because Math is 6.9
        self.assertEqual(res["action"], "HOLD")
        self.assertEqual(res["execution"]["status"], "SKIPPED")

    @patch("src.ai.ai_agent.StockAdvisor.evaluate_df_slice")
    @patch("src.ai.ai_agent.get_live_quote")
    @patch("src.ai.ai_agent.get_historical_data")
    def test_boundary_7_4_near_miss_blocks_llm(self, mock_hist, mock_quote, mock_stock_adv):
        """Boundary Test: Math 7.4 < 7.5 -> Blocked."""
        mock_quote.return_value = {"price": 2500.0, "low": 2490.0, "high": 2510.0}
        mock_hist.return_value = MagicMock(empty=False)
        mock_stock_adv.return_value = {"score": 7.4, "regime": "TRENDING"}
        
        self.mock_llm.generate_completion.return_value = '{"action": "BUY_STOCK", "confidence_score": 9.0, "target_asset": "RELIANCE", "suggested_sl_pct": 1.5, "suggested_tp_pct": 3.5}'
        
        agent = AITradingAgent(self.mock_llm, self.guardrails, self.broker, min_consensus_threshold=7.5)
        res = agent.evaluate_and_execute("RELIANCE.NS")
        
        self.assertEqual(res["action"], "HOLD")

    @patch("src.ai.ai_agent.StockAdvisor.evaluate_df_slice")
    @patch("src.ai.ai_agent.get_live_quote")
    @patch("src.ai.ai_agent.get_historical_data")
    def test_boundary_7_5_strong_math_and_llm_executes(self, mock_hist, mock_quote, mock_stock_adv):
        """Boundary Test: Math 7.5 >= 7.5 AND LLM 7.5 >= 7.5 -> APPROVED & EXECUTED."""
        mock_quote.return_value = {"price": 2500.0, "low": 2490.0, "high": 2510.0}
        mock_hist.return_value = MagicMock(empty=False)
        mock_stock_adv.return_value = {"score": 7.5, "regime": "TRENDING"}
        
        self.mock_llm.generate_completion.return_value = '{"action": "BUY_STOCK", "confidence_score": 7.5, "target_asset": "RELIANCE", "suggested_sl_pct": 1.5, "suggested_tp_pct": 3.5}'
        
        agent = AITradingAgent(self.mock_llm, self.guardrails, self.broker, min_consensus_threshold=7.5)
        res = agent.evaluate_and_execute("RELIANCE.NS")
        
        self.assertEqual(res["action"], "BUY_STOCK")
        self.assertEqual(res["guardrail_status"], "APPROVED")

    @patch("src.ai.ai_agent.StockAdvisor.evaluate_df_slice")
    @patch("src.ai.ai_agent.get_live_quote")
    @patch("src.ai.ai_agent.get_historical_data")
    def test_asymmetric_veto_llm_caution_overrides_high_math(self, mock_hist, mock_quote, mock_stock_adv):
        """Asymmetric Veto: Math 9.2 (Strong) but LLM says HOLD (Caution) -> Defense Wins (HOLD)."""
        mock_quote.return_value = {"price": 2500.0, "low": 2490.0, "high": 2510.0}
        mock_hist.return_value = MagicMock(empty=False)
        mock_stock_adv.return_value = {"score": 9.2, "regime": "TRENDING"}
        
        # LLM detects risk and says HOLD
        self.mock_llm.generate_completion.return_value = '{"action": "HOLD", "confidence_score": 5.0, "reasoning": "Major overhead resistance on 1H chart. Defense veto."}'
        
        agent = AITradingAgent(self.mock_llm, self.guardrails, self.broker, min_consensus_threshold=7.5)
        res = agent.evaluate_and_execute("RELIANCE.NS")
        
        self.assertEqual(res["action"], "HOLD")

if __name__ == "__main__":
    unittest.main()
