"""
Automated Test Suite for AI Market Opportunity Radar and Multi-Asset Scanner.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from src.ai.market_radar import MarketRadarScanner
from src.ai.llm_client import LLMClient
from src.ai.ai_agent import AITradingAgent
from src.engine.ai_guardrails import AIGuardrails
from src.brokers.paper_broker import PaperBroker
from src.utils.storage import reset_all_data

class TestAIMarketRadar(unittest.TestCase):
    
    def setUp(self):
        reset_all_data(100000.0)
        self.mock_llm = MagicMock(spec=LLMClient)
        self.mock_llm.provider = "gemini"
        self.mock_llm.model = "gemini-3.1-flash-lite"
        self.guardrails = AIGuardrails(max_daily_loss_flat=2000.0, min_confidence_threshold=7.5)
        self.broker = PaperBroker(initial_capital=100000)
        self.agent = AITradingAgent(
            llm_client=self.mock_llm,
            guardrails=self.guardrails,
            broker=self.broker,
            is_live_mode=False
        )

    @patch("src.ai.market_radar.get_historical_data")
    @patch("src.ai.market_radar.get_live_quote")
    def test_radar_scanner_parses_opportunities(self, mock_quote, mock_hist):
        mock_quote.return_value = {
            "symbol": "NIFTY",
            "price": 24500.0,
            "change_pct": 0.85,
            "high": 24550.0,
            "low": 24420.0,
            "volume": 1200000
        }
        dates = pd.date_range("2026-01-01", periods=30, freq="5min")
        mock_hist.return_value = pd.DataFrame({
            "Open": np.linspace(24400, 24500, 30),
            "High": np.linspace(24410, 24520, 30),
            "Low": np.linspace(24390, 24490, 30),
            "Close": np.linspace(24405, 24500, 30),
            "Volume": [10000]*30
        }, index=dates)

        sample_json_response = """
        {
          "market_summary": "Nifty in strong uptrend above 20 EMA, BankNifty consolidating.",
          "scanned_count": 8,
          "opportunities": [
            {
              "rank": 1,
              "symbol": "NIFTY",
              "instrument_type": "INDEX_OPTION",
              "option_contract": "NIFTY 24500 CE",
              "action": "BUY_CALL",
              "setup_name": "5m EMA Breakout & VWAP Rebound",
              "time_horizon": "30-45 mins (Intraday Scalp)",
              "entry_price": 145.0,
              "stop_loss": 125.0,
              "target_1": 185.0,
              "target_2": 215.0,
              "risk_reward_ratio": "1:2.5",
              "expected_gain_pct": "+27% to +48%",
              "confidence_score": 8.8,
              "catalyst_reasoning": "Strong call unwinding and spot broke above day VWAP."
            }
          ]
        }
        """
        self.mock_llm.generate_completion.return_value = sample_json_response
        
        res = MarketRadarScanner.scan_market(
            llm_client=self.mock_llm,
            min_confidence=7.5
        )
        
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(len(res["opportunities"]), 1)
        top_opp = res["opportunities"][0]
        self.assertEqual(top_opp["symbol"], "NIFTY")
        self.assertEqual(top_opp["action"], "BUY_CALL")
        self.assertEqual(top_opp["time_horizon"], "30-45 mins (Intraday Scalp)")
        self.assertEqual(top_opp["confidence_score"], 8.8)

    def test_execute_radar_opportunity_through_guardrails(self):
        opp = {
            "symbol": "NIFTY",
            "action": "BUY_CALL",
            "option_contract": "NIFTY 24500 CE",
            "entry_price": 145.0,
            "stop_loss": 125.0,
            "target_1": 185.0,
            "confidence_score": 8.5,
            "time_horizon": "30-45 mins (Intraday)",
            "setup_name": "5m EMA Breakout",
            "catalyst_reasoning": "Breakout with high volume"
        }
        
        exec_res = self.agent.execute_radar_opportunity(opp)
        self.assertEqual(exec_res["status"], "EXECUTED")
        self.assertEqual(len(self.broker.get_open_positions()), 1)
        self.assertEqual(self.broker.get_open_positions()[0]["symbol"], "NIFTY 24500 CE")

if __name__ == "__main__":
    unittest.main()
