"""
Unit tests for AI Assistant Enhancements:
1. Agentic Tool Calling
2. Multi-Turn Context Memory & Pronoun Resolution
3. Mini-Chart Payload Generation & Price Ladders
"""

import unittest
from unittest.mock import MagicMock
from src.ai.assistant_tools import AssistantToolRunner
from src.ai.chat_assistant import TradingChatAssistant
from src.brokers.paper_broker import PaperBroker
from src.utils.helpers import get_ist_now

class TestAIAssistantEnhancements(unittest.TestCase):

    def setUp(self):
        self.broker = PaperBroker(initial_capital=100000.0)

    def test_portfolio_tool(self):
        """Tests that the portfolio tool accurately calculates cash and returns PORTFOLIO card."""
        res = AssistantToolRunner.get_portfolio_status(self.broker)
        self.assertTrue(res["success"])
        self.assertEqual(res["ui_card_type"], "PORTFOLIO")
        self.assertIn("cash", res["data"])
        self.assertGreaterEqual(res["data"]["cash"], 0.0)
        self.assertIn("Live Portfolio Status", res["summary_markdown"])

    def test_options_recommendation_tool(self):
        """Tests Black-Scholes Greeks options strike selector tool."""
        res = AssistantToolRunner.get_options_recommendation("NIFTY", bias="BUY_CALL", dte_days=3.0)
        self.assertTrue(res["success"])
        self.assertEqual(res["ui_card_type"], "OPTIONS")
        data = res["data"]
        self.assertIn("CE", data["contract_symbol"])
        self.assertGreater(data["theoretical_premium"], 0.0)
        self.assertIn("delta", data["greeks"])
        self.assertGreater(data["greeks"]["delta"], 0.0)

    def test_square_off_tool_empty_positions(self):
        """Tests square off tool when there are no open positions."""
        mock_broker = MagicMock()
        mock_broker.get_open_positions.return_value = []
        res = AssistantToolRunner.square_off_action("Reliance", mock_broker)
        self.assertFalse(res["success"])
        self.assertIn("No Open Positions Found", res["summary_markdown"])

    def test_multi_turn_context_resolution(self):
        """Tests conversational memory and follow-up query understanding."""
        # 1. First turn: Analyze Tata Motors (resolves to TMCV.NS)
        res1 = TradingChatAssistant.process_query(
            user_query="How is Tata Motors looking for intraday?",
            broker_instance=self.broker
        )
        ctx = res1.get("updated_context")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.get("last_symbol"), "TMCV.NS")
        self.assertEqual(ctx.get("last_horizon"), "intraday")

        # 2. Second turn: Follow up with "What about for swing trading?" (No stock mentioned)
        res2 = TradingChatAssistant.process_query(
            user_query="What about for swing trading?",
            active_context=ctx,
            broker_instance=self.broker
        )
        self.assertEqual(res2["symbol"], "TMCV.NS")
        self.assertEqual(res2["updated_context"]["last_horizon"], "swing")

        # 3. Third turn: Follow up with "Where is the stop loss?"
        res3 = TradingChatAssistant.process_query(
            user_query="Where is the stop loss?",
            active_context=res2["updated_context"],
            broker_instance=self.broker
        )
        self.assertEqual(res3["symbol"], "TMCV.NS")

    def test_mini_chart_payload(self):
        """Tests that stock analysis generates a complete mini-chart payload."""
        res = TradingChatAssistant.process_query(
            user_query="Analyze Reliance",
            broker_instance=self.broker
        )
        chart = res.get("chart_data")
        self.assertIsNotNone(chart)
        self.assertEqual(chart["symbol"], "RELIANCE.NS")
        self.assertGreater(len(chart["dates"]), 0)
        self.assertEqual(len(chart["dates"]), len(chart["close"]))
        self.assertGreater(chart["entry_price"], 0.0)
        self.assertGreater(chart["target_1"], 0.0)
        self.assertGreater(chart["stop_loss"], 0.0)

    def test_json_response_auto_formatter(self):
        """Tests that raw JSON responses from LLMs are converted into beautiful Markdown."""
        raw_json = '''
        {
            "stock": "IZMO Ltd",
            "analysis_summary": "IZMO is showing strong small-cap momentum.",
            "live_price_and_score": { "live_price": "₹485.50", "mathematical_score": "7.2/10" },
            "trade_plan": {
                "ideal_entry_zone": "₹475 - ₹482",
                "target_1": { "price": "₹515", "gain_rs": "33", "gain_percent": "6.8%", "action": "Lock 50% profits" },
                "target_2": { "price": "₹550", "gain_rs": "68", "gain_percent": "14.1%", "action": "Runner" },
                "safety_stop_loss": { "price": "₹458", "loss_rs": "24", "loss_percent": "5.0%", "action": "Mandatory exit" }
            },
            "risk_reward_ratio": "1.4:1",
            "risk_management_note": "Volatile small cap stock."
        }
        '''
        formatted_md, data = TradingChatAssistant._parse_and_format_json_response(raw_json)
        self.assertIsNotNone(data)
        self.assertIn("Institutional Analysis for IZMO Ltd", formatted_md)
        self.assertIn("Target 1", formatted_md)
        self.assertIn("Safety Stop-Loss", formatted_md)
        self.assertNotIn('{"stock"', formatted_md)

    def test_concatenated_market_analysis_json_formatting(self):
        """Tests that concatenated/duplicated { 'market_analysis': { ... } } responses are parsed and formatted cleanly."""
        raw_concat = '''{ "market_analysis": { "ticker": "RELIANCE", "live_price": "2845.50", "mathematical_score": "8.2/10", "trade_plan": { "entry_zone": "2835.00 - 2842.00", "target_1": { "price": "2875.00", "gain_rs": "+29.50", "gain_pct": "+1.04%", "action": "Lock 50% profit, move SL to Breakeven" }, "target_2": { "price": "2910.00", "gain_rs": "+64.50", "gain_pct": "+2.27%", "action": "Runner" }, "stop_loss": { "price": "2818.00", "loss_rs": "-27.00", "loss_pct": "-0.95%", "action": "Mandatory exit" }, "risk_reward_ratio": "2.38:1" }, "technical_notes": "Stock is holding above the 20-day EMA with rising volume." } }{ "market_analysis": { "ticker": "RELIANCE", "live_price": "2845.50" } }'''
        formatted_md, data = TradingChatAssistant._parse_and_format_json_response(raw_concat)
        self.assertIsNotNone(data)
        self.assertIn("Institutional Analysis for RELIANCE", formatted_md)
        self.assertIn("2875.00", formatted_md)
        self.assertIn("2818.00", formatted_md)
        self.assertIn("Lock 50% profit", formatted_md)
        self.assertNotIn('{"market_analysis"', formatted_md)

    def test_json_wrapper_unnesting(self):
        """Tests that JSON responses wrapped in {"response": "..."} are unnested cleanly into Markdown."""
        wrapper_json = '''
        {
            "response": "### 🎯 ApexTrade AI: Daily Market Guidance\\n\\nHere are the recommended trading rules for today."
        }
        '''
        formatted_md, data = TradingChatAssistant._parse_and_format_json_response(wrapper_json)
        self.assertIn("Daily Market Guidance", formatted_md)
        self.assertNotIn('{"response":', formatted_md)

    def test_top_picks_query(self):
        """Tests that queries like 'suggest me what will i buy for today ?' trigger high-conviction recommendations."""
        mock_picks = [{
            "symbol": "RELIANCE.NS", "display_name": "Reliance", "current_price": 2500.0, "score": 8.5,
            "setup_grade_title": "GRADE A", "action": "BUY",
            "target_1": {"price": 2575.0, "gain_pct": 3.0},
            "target_2": {"price": 2650.0, "gain_pct": 6.0},
            "stop_loss": {"price": 2450.0, "loss_pct": 2.0},
            "levels": {"risk_reward": "1:2.0"}
        }]
        res = TradingChatAssistant.process_query(
            user_query="suggest me what will i buy for today ?",
            broker_instance=self.broker,
            last_scanned_picks=mock_picks
        )
        self.assertTrue(len(res["response_text"]) > 20)
        self.assertNotIn('{"response":', res["response_text"])

    def test_camarilla_pivots(self):
        """Tests 8-level institutional Camarilla pivot equations."""
        from src.strategies.indicators import calculate_camarilla_pivots
        pivots = calculate_camarilla_pivots(high=2550.0, low=2480.0, close=2520.0)
        self.assertIn("h4", pivots)
        self.assertIn("l4", pivots)
        self.assertIn("h3", pivots)
        self.assertIn("l3", pivots)
        self.assertGreater(pivots["h4"], pivots["h3"])
        self.assertGreater(pivots["h3"], pivots["h2"])
        self.assertGreater(pivots["l2"], pivots["l3"])
        self.assertGreater(pivots["l3"], pivots["l4"])

    def test_volume_profile_poc(self):
        """Tests Volume Profile POC and Value Area (VAH/VAL) extraction."""
        import pandas as pd
        from src.strategies.indicators import calculate_volume_profile
        data = {
            "Open": [100, 102, 105, 104, 106, 108, 107, 109, 110, 108],
            "High": [103, 106, 107, 106, 109, 110, 111, 112, 112, 111],
            "Low": [99, 101, 103, 102, 105, 106, 106, 108, 108, 107],
            "Close": [102, 105, 104, 106, 108, 107, 109, 110, 108, 110],
            "Volume": [1000, 1500, 5000, 2000, 1200, 800, 1100, 950, 1300, 1400]
        }
        df = pd.DataFrame(data)
        vp = calculate_volume_profile(df, bins=10)
        self.assertGreater(vp["poc"], 0.0)
        self.assertGreaterEqual(vp["vah"], vp["val"])
        self.assertIn(vp["location"], ["ABOVE_VALUE_PREMIUM", "BELOW_VALUE_DISCOUNT", "INSIDE_FAIR_VALUE"])

    def test_hurst_exponent(self):
        """Tests Hurst Exponent calculation."""
        import pandas as pd
        import numpy as np
        from src.strategies.indicators import calculate_hurst_exponent
        prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100))))
        h = calculate_hurst_exponent(prices)
        self.assertGreaterEqual(h, 0.05)
        self.assertLessEqual(h, 0.95)

    def test_stock_advisor_institutional_payload(self):
        """Tests that StockAdvisor includes all institutional quant metrics."""
        from src.engine.stock_advisor import StockAdvisor
        res = StockAdvisor.analyze_stock("RELIANCE.NS", horizon="intraday")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("camarilla_pivots", res)
        self.assertIn("volume_profile", res)
        self.assertIn("hurst_exponent", res)
        self.assertIn("fvg_structure", res)
        self.assertIn("ttm_squeeze", res)

    def test_technical_scanner_golden_cross(self):
        """Tests that run_technical_scanner executes golden cross scans with exact calculations."""
        res = AssistantToolRunner.run_technical_scanner(scan_type="golden_cross")
        self.assertTrue(res["success"])
        self.assertEqual(res["ui_card_type"], "SCREENER")
        self.assertIn("Golden Cross", res["summary_markdown"])

    def test_golden_cross_chat_query(self):
        """Tests that natural language queries for Golden Cross trigger direct quantitative scans."""
        res = TradingChatAssistant.process_query(
            user_query="scan golden cross stocks for nifty",
            broker_instance=self.broker
        )
        self.assertEqual(res["ui_card_type"], "SCREENER")
        self.assertIn("Golden Cross", res["response_text"])
        self.assertNotIn("While I am an AI assistant and not a real-time terminal", res["response_text"])

    def test_disclaimer_stripper(self):
        """Tests that robotic AI disclaimers are completely stripped from responses."""
        bad_text = "While I am an AI assistant and not a real-time terminal, I have analyzed Nifty 500 stocks. Here are the results."
        cleaned = TradingChatAssistant._clean_disclaimers(bad_text)
        self.assertNotIn("While I am an AI assistant and not a real-time terminal", cleaned)
        self.assertIn("Here are the results", cleaned)

if __name__ == "__main__":
    unittest.main()
