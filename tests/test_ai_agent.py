"""
Comprehensive Automated Test Suite for Autonomous AI Trading Agent, Guardrails, and Zerodha Adapter.
"""

import unittest
import json
import time
from src.ai.llm_client import LLMClient
from src.ai.failsafe import FailsafeParser
from src.ai.market_prompter import MarketPrompter
from src.ai.calibration import ConfidenceCalibrator
from src.ai.ai_agent import AITradingAgent
from src.engine.ai_guardrails import AIGuardrails
from src.engine.reconciliation import StateReconciler
from src.brokers.paper_broker import PaperBroker
from src.brokers.zerodha_live import ZerodhaLiveBroker
from src.backtest.ai_backtester import AIBacktester

class TestAITradingAgentSuite(unittest.TestCase):

    def test_failsafe_parser_corrupted_json(self):
        """Verify that malformed JSON or hallucinations strictly default to HOLD."""
        # 1. Broken JSON
        bad_json = "{ action: BUY_CALL, confidence_score: 'ten' broken }"
        res = FailsafeParser.parse_and_validate(bad_json)
        self.assertEqual(res["action"], "HOLD")
        self.assertTrue(res["is_failsafe"])
        
        # 2. Empty string
        res_empty = FailsafeParser.parse_and_validate("")
        self.assertEqual(res_empty["action"], "HOLD")
        self.assertTrue(res_empty["is_failsafe"])
        
        # 3. Invalid action
        fake_action = json.dumps({"action": "YOLO_CALLS", "confidence_score": 9.9})
        res_fake = FailsafeParser.parse_and_validate(fake_action)
        self.assertEqual(res_fake["action"], "HOLD")

    def test_failsafe_parser_valid_json(self):
        """Verify valid structured JSON parses correctly."""
        valid_json = json.dumps({
            "action": "BUY_CALL",
            "target_asset": "NIFTY",
            "strike_offset": "ATM",
            "confidence_score": 8.5,
            "reasoning": "Strong bullish EMA breakout with RSI > 60.",
            "suggested_sl_pct": 1.5,
            "suggested_tp_pct": 3.0,
            "risk_level": "LOW"
        })
        res = FailsafeParser.parse_and_validate(valid_json)
        self.assertEqual(res["action"], "BUY_CALL")
        self.assertEqual(res["confidence_score"], 8.5)
        self.assertEqual(res["target_asset"], "NIFTY")
        self.assertFalse(res["is_failsafe"])

    def test_guardrail_confidence_filter(self):
        """Verify trade proposal is blocked if AI confidence is below threshold."""
        guardrails = AIGuardrails(min_confidence_threshold=7.5)
        proposal = {
            "action": "BUY_CALL",
            "target_asset": "NIFTY",
            "confidence_score": 6.8, # Below 7.5
            "suggested_sl_pct": 1.5,
            "suggested_tp_pct": 3.0
        }
        approved, reason, sanitized = guardrails.evaluate_proposal(proposal, {"capital": 100000, "daily_pnl": 0})
        self.assertFalse(approved)
        self.assertIn("Confidence", reason)

    def test_guardrail_daily_loss_circuit_breaker(self):
        """Verify bot halts immediately if max daily drawdown is hit."""
        guardrails = AIGuardrails(max_daily_loss_flat=2000.0, max_daily_loss_pct=3.0)
        proposal = {
            "action": "BUY_CALL",
            "target_asset": "NIFTY",
            "confidence_score": 9.0,
            "suggested_sl_pct": 1.5,
            "suggested_tp_pct": 3.0
        }
        # Portfolio down by ₹2,500
        portfolio = {"capital": 100000, "daily_pnl": -2500.0, "open_positions": []}
        approved, reason, sanitized = guardrails.evaluate_proposal(proposal, portfolio)
        self.assertFalse(approved)
        self.assertIn("Drawdown", reason)

    def test_guardrail_revenge_trading_cooldown(self):
        """Verify post-SL cooldown blocks immediate revenge trade."""
        guardrails = AIGuardrails(sl_cooldown_minutes=15)
        guardrails.register_stop_loss_hit("NIFTY")
        
        proposal = {
            "action": "BUY_CALL",
            "target_asset": "NIFTY",
            "confidence_score": 8.5,
            "suggested_sl_pct": 1.5,
            "suggested_tp_pct": 3.0
        }
        approved, reason, sanitized = guardrails.evaluate_proposal(proposal, {"capital": 100000, "daily_pnl": 0})
        self.assertFalse(approved)
        self.assertIn("Cooldown Active", reason)

    def test_guardrail_illiquid_bid_ask_spread(self):
        """Verify wide bid-ask spread contracts are rejected."""
        guardrails = AIGuardrails(max_bid_ask_spread_pct=2.5)
        proposal = {
            "action": "BUY_CALL",
            "target_asset": "NIFTY",
            "confidence_score": 8.5,
            "suggested_sl_pct": 1.5,
            "suggested_tp_pct": 3.0
        }
        # Spread is 5.0% (Bid: 95, Ask: 100, LTP: 100)
        market_depth = {"price": 100.0, "bid": 95.0, "ask": 100.0}
        approved, reason, sanitized = guardrails.evaluate_proposal(proposal, {"capital": 100000, "daily_pnl": 0}, market_depth)
        self.assertFalse(approved)
        self.assertIn("Illiquid", reason)

    def test_zerodha_live_option_symbol_resolver(self):
        """Verify Zerodha live broker resolves standard NFO symbols."""
        broker = ZerodhaLiveBroker(api_key="test", api_secret="test")
        sym = broker.resolve_option_symbol("NIFTY", strike=24500, option_type="CE")
        self.assertTrue(sym.startswith("NIFTY"))
        self.assertTrue(sym.endswith("24500CE"))

    def test_state_reconciliation_ground_truth(self):
        """Verify state reconciler correctly pulls positions from broker."""
        paper = PaperBroker(initial_capital=100000.0)
        paper.square_off_all(reason="Test Reset")
        paper.place_order(symbol="RELIANCE.NS", side="BUY", quantity=10, price=1300.0)
        
        reconciled = StateReconciler.reconcile_with_broker(paper)
        self.assertEqual(reconciled["status"], "SYNCED")
        self.assertEqual(reconciled["active_legs_count"], 1)
        self.assertEqual(reconciled["open_positions"][0]["symbol"], "RELIANCE.NS")

    def test_ai_backtester_replay(self):
        """Verify AI backtester runs across market regimes without error."""
        res = AIBacktester.run_regime_backtest(symbol="RELIANCE.NS", regime="BULL_TREND", sample_bars=25)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("win_rate", res)
        self.assertIn("equity_curve", res)

if __name__ == "__main__":
    unittest.main()
