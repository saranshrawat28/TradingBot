"""
Unit Test Suite for Conversational AI Trading Assistant & Guardrail Integration.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.ai.chat_assistant import TradingChatAssistant
from src.engine.ai_guardrails import AIGuardrails

class TestTradingChatAssistant(unittest.TestCase):

    def test_resolve_symbol_from_natural_language(self):
        """Verify natural language ticker resolution for common Indian stocks."""
        self.assertEqual(TradingChatAssistant.resolve_symbol_from_text("How is Tata Motors looking?"), "TMCV.NS")
        self.assertEqual(TradingChatAssistant.resolve_symbol_from_text("Analyze Reliance for intraday"), "RELIANCE.NS")
        self.assertEqual(TradingChatAssistant.resolve_symbol_from_text("What about INFY?"), "INFY.NS")

    @patch("src.ai.chat_assistant.PreMarketAnalyzer.get_market_opening_sentiment")
    def test_process_query_market_sentiment(self, mock_sentiment):
        """Verify market overview intent queries PreMarketAnalyzer."""
        mock_sentiment.return_value = {
            "title": "🟢 BULLISH OPEN EXPECTED",
            "explanation": "Nifty is opening with +0.50% gap up.",
            "nifty_price": 24500.0,
            "gap_pct": 0.50
        }
        res = TradingChatAssistant.process_query("What is the market opening mood today?")
        self.assertIn("response_text", res)
        self.assertIn("BULLISH OPEN", res["response_text"])
        self.assertTrue(res["is_local_fallback"])

    @patch("src.ai.chat_assistant.PreMarketAnalyzer.scan_pre_market_stocks")
    def test_process_query_top_picks(self, mock_scan):
        """Verify top picks intent returns curated morning recommendations."""
        mock_scan.return_value = {
            "top_picks": [
                {
                    "symbol": "TMCV.NS",
                    "display_name": "TATA MOTORS",
                    "action_title": "🟢 STRONG BUY",
                    "current_price": 1000.0,
                    "target_1_price": 1030.0,
                    "target_1_gain_pct": 3.0,
                    "stop_loss_price": 985.0
                }
            ]
        }
        res = TradingChatAssistant.process_query("Show me the best stocks to buy today")
        self.assertIn("TATA MOTORS", res["response_text"])
        self.assertIn("1,030.00", res["response_text"])

    @patch("src.ai.chat_assistant.StockAdvisor.analyze_stock")
    def test_process_query_trade_proposal_action_card(self, mock_analyze):
        """Verify trade intent generates an explicit, deterministic Action Card."""
        mock_analyze.return_value = {
            "status": "SUCCESS",
            "display_name": "RELIANCE",
            "current_price": 2500.0,
            "score": 8.2,
            "verdict": "STRONG_BUY",
            "verdict_desc": "Strong momentum breakout.",
            "target_1": {"price": 2575.0, "gain_pct": 3.0},
            "stop_loss": {"price": 2450.0, "loss_pct": 2.0}
        }
        res = TradingChatAssistant.process_query("Buy ₹50,000 of Reliance with stop-loss")
        self.assertIsNotNone(res["action_card"])
        card = res["action_card"]
        self.assertEqual(card["symbol"], "RELIANCE.NS")
        self.assertEqual(card["action"], "BUY")
        self.assertEqual(card["quantity"], 20) # 50,000 / 2500 = 20 shares
        self.assertEqual(card["capital_required"], 50000.0)
        self.assertEqual(card["target_1_price"], 2575.0)
        self.assertEqual(card["stop_loss_price"], 2450.0)

    def test_guardrail_zero_bypass_on_chat_action_card(self):
        """Verify that an action card proposal strictly executes through AIGuardrails without exception."""
        # Simulated chat action card
        card = {
            "symbol": "RELIANCE.NS",
            "action": "BUY",
            "quantity": 10,
            "entry_price": 2500.0,
            "sl_price": 2450.0,
            "t1_price": 2575.0,
            "t2_price": 2625.0,
            "score": 8.5,
            "reason": "Chat order"
        }

        proposal = {
            "symbol": card["symbol"],
            "target_asset": card["symbol"],
            "action": "BUY_STOCK",
            "confidence_score": card["score"],
            "entry_price": card["entry_price"],
            "sl": card["sl_price"],
            "target_1": card["t1_price"],
            "target_2": card["t2_price"],
            "horizon": "intraday",
            "notes": card["reason"]
        }

        # Case A: Normal state -> Approved
        portfolio_state = {"cash": 100000.0, "daily_pnl": 0.0, "daily_drawdown_limit": 2000.0, "positions": {}}
        guard = AIGuardrails(min_confidence_threshold=7.0)
        approved, reason, sanitized = guard.evaluate_proposal(proposal, portfolio_state, enforce_time_cutoff=False)
        self.assertTrue(approved)
        self.assertIn("quantity", sanitized)

        # Case B: Max Daily Drawdown breached -> Strictly BLOCKED
        drawdown_state = {"cash": 95000.0, "daily_pnl": -2500.0, "daily_drawdown_limit": 2000.0, "positions": {}}
        approved_dd, reason_dd, _ = guard.evaluate_proposal(proposal, drawdown_state, enforce_time_cutoff=False)
        self.assertFalse(approved_dd)
        self.assertIn("Drawdown Limit Hit", reason_dd)

if __name__ == "__main__":
    unittest.main()
