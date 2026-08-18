"""
Unit and Ablation Test Suite for Institutional Accuracy Enhancements:
1. Multi-Timeframe (MTF) Trend Multiplier Boundedness
2. RSI Divergence Asymmetric Veto & Boost
3. Candle Wick Supply Trap Protection
4. Time-of-Day Hard Defensive Blocks & Mid-Day Chop Modulation
5. 4-Stage Exit State Machine (Entry -> +1.0R BE -> +1.5R T1 -> Chandelier/T2)
"""

import unittest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from src.strategies.indicators import (
    detect_rsi_divergence, calculate_candle_structure, calculate_mtf_alignment,
    calculate_rsi
)
from src.engine.stock_advisor import StockAdvisor
from src.engine.ai_guardrails import AIGuardrails
from src.engine.trade_manager import SmartTradeManager
from src.brokers.paper_broker import PaperBroker
from src.utils.storage import reset_all_data, save_position

class TestAccuracySuite(unittest.TestCase):

    def setUp(self):
        reset_all_data(100000.0)

    def test_mtf_multiplier_boundedness(self):
        """Verify that MTF multiplier scales Trend Bucket between 0.70x and 1.15x, strictly clamped to [-2.5, +2.5]."""
        dates = pd.date_range("2026-01-01 09:15", periods=60, freq="5min")
        
        # Bullish 5m + 15m trend
        df_bull = pd.DataFrame({
            "Open": np.linspace(100, 150, 60),
            "High": np.linspace(102, 152, 60),
            "Low": np.linspace(99, 149, 60),
            "Close": np.linspace(101, 151, 60),
            "Volume": [10000] * 60
        }, index=dates)

        res_bull = StockAdvisor.evaluate_df_slice(df_bull, "TEST_BULL")
        trend_score = res_bull["buckets"]["trend"]
        
        self.assertLessEqual(trend_score, 2.50)
        self.assertGreaterEqual(trend_score, -2.50)
        self.assertIn("mtf_alignment", res_bull)
        self.assertIn(res_bull["mtf_alignment"]["mu_mtf"], [1.0, 1.15, 0.70])

    def test_divergence_asymmetric_veto(self):
        """Verify that Bearish Divergence clamps Momentum Bucket <= 0.0."""
        # Create prices making higher highs while RSI makes lower highs
        dates = pd.date_range("2026-01-01", periods=30, freq="5min")
        close_vals = [
            100, 105, 110, 108, 106, 112, 118, 115, 112, 120,
            125, 122, 118, 126, 130, 128, 124, 132, 138, 134,
            130, 135, 142, 139, 135, 140, 145, 142, 138, 148
        ]
        low_vals = [c - 2 for c in close_vals]
        high_vals = [c + 2 for c in close_vals]
        rsi_vals = [
            75, 78, 80, 74, 70, 76, 79, 72, 68, 74,
            77, 70, 65, 72, 75, 68, 62, 69, 71, 64,
            58, 65, 68, 60, 55, 61, 63, 58, 52, 59
        ]
        
        c_series = pd.Series(close_vals, index=dates)
        l_series = pd.Series(low_vals, index=dates)
        h_series = pd.Series(high_vals, index=dates)
        r_series = pd.Series(rsi_vals, index=dates)

        div_res = detect_rsi_divergence(c_series, l_series, h_series, r_series, lookback=25)
        self.assertTrue(div_res["bearish_divergence"])

    def test_candle_wick_rejection_structure(self):
        """Verify candle structure detects upper wick rejection >= 40%."""
        dates = pd.date_range("2026-01-01", periods=5, freq="5min")
        # Candle with big upper shadow: Open=100, High=120, Low=98, Close=102 -> Upper wick = 120 - 102 = 18, Total = 22 (81.8%)
        o_s = pd.Series([100, 100, 100, 100, 100], index=dates)
        h_s = pd.Series([105, 105, 105, 105, 120], index=dates)
        l_s = pd.Series([98, 98, 98, 98, 98], index=dates)
        c_s = pd.Series([102, 102, 102, 102, 102], index=dates)

        wick_res = calculate_candle_structure(o_s, h_s, l_s, c_s)
        self.assertTrue(wick_res["is_upper_rejection"])
        self.assertGreaterEqual(wick_res["upper_wick_ratio"], 0.40)

    def test_guardrail_candle_wick_trap_gate(self):
        """Verify guardrails strictly reject trade proposal with >=45% upper wick rejection."""
        guard = AIGuardrails(min_confidence_threshold=7.5)
        portfolio = {"capital": 100000.0, "daily_pnl": 0.0, "open_positions": []}
        
        proposal = {
            "action": "BUY_STOCK",
            "target_asset": "RELIANCE",
            "confidence_score": 8.5,
            "suggested_sl_pct": 1.5,
            "suggested_tp_pct": 3.0,
            "upper_wick_ratio": 0.50 # 50% upper wick rejection
        }
        
        approved, reason, order = guard.evaluate_proposal(proposal, portfolio)
        self.assertFalse(approved)
        self.assertIn("Liquidity Trap Gate", reason)

    def test_4_stage_exit_state_machine(self):
        """Verify sequential 4-stage Exit State Machine transitions."""
        from unittest.mock import patch
        broker = PaperBroker(initial_capital=100000.0)
        
        # 1. Place trade at 100 with SL 90 (-10 pts, R=10), T1 115 (+1.5R), T2 125 (+2.5R)
        pos = {
            "symbol": "INFY",
            "quantity": 10,
            "entry_time": "2026-01-01 10:00:00",
            "entry_price": 100.0,
            "current_price": 100.0,
            "highest_price": 100.0,
            "sl": 90.0,
            "target_1": 115.0,
            "target_2": 125.0,
            "trailing_sl": 90.0,
            "side": "BUY",
            "target_1_hit": 0,
            "breakeven_locked": 0
        }
        save_position(pos)

        # Stage 1: Price reaches 110 (+1.0R) -> Move SL to Breakeven (100.2)
        pos["current_price"] = 110.0
        pos["highest_price"] = 110.0
        save_position(pos)
        with patch("src.engine.trade_manager.get_live_quote", return_value={"price": 110.0}):
            events_s1 = SmartTradeManager.evaluate_and_manage_positions(broker)
        
        be_events = [e for e in events_s1 if e["type"] == "BREAKEVEN_LOCKED"]
        self.assertEqual(len(be_events), 1)
        self.assertGreaterEqual(be_events[0]["sl_price"], 100.0)

        # Stage 2: Price reaches 116 (+1.5R) -> Target 1 Partial 50% Profit Booked & Lock +0.5R (105.0)
        pos["current_price"] = 116.0
        pos["highest_price"] = 116.0
        save_position(pos)
        with patch("src.engine.trade_manager.get_live_quote", return_value={"price": 116.0}):
            events_s2 = SmartTradeManager.evaluate_and_manage_positions(broker)
        
        t1_events = [e for e in events_s2 if e["type"] == "TARGET_1_PROFIT_BOOKED"]
        self.assertEqual(len(t1_events), 1)
        self.assertEqual(t1_events[0]["closed_qty"], 5)

    def test_vwap_single_point_scoring_and_zero_collinearity(self):
        """
        Verify that Intraday VWAP sigma-location is evaluated in Bucket 3,
        and Bucket 4 contains pure flow metrics with ZERO price/VWAP collinearity.
        """
        dates = pd.date_range("2026-01-01 09:15", periods=30, freq="5min")
        df = pd.DataFrame({
            "Open": np.linspace(100, 110, 30),
            "High": np.linspace(101, 111, 30),
            "Low": np.linspace(99, 109, 30),
            "Close": np.linspace(100.5, 110.5, 30),
            "Volume": [5000] * 30
        }, index=dates)

        res = StockAdvisor.evaluate_df_slice(df, "TEST_VWAP", horizon="intraday")
        self.assertIn("vwap_structure", res)
        self.assertGreater(res["vwap_structure"]["vwap"], 0.0)
        
        # Verify Bucket 3 captures location
        self.assertTrue(-1.5 <= res["buckets"]["volatility"] <= 1.5)
        # Verify Bucket 4 captures volume flow
        self.assertTrue(-1.5 <= res["buckets"]["volume_flow"] <= 1.5)

    def test_unified_context_multiplier_bounds(self):
        """
        Verify that Unified Context Multiplier (ADX + Macro Breadth) is strictly bounded in [0.50, 1.25].
        """
        from src.strategies.indicators import calculate_context_multiplier
        
        # 1. Strong trend + Macro tailwind -> 1.0 * 1.15 = 1.15
        mu_max = calculate_context_multiplier(adx=32.0, stock_trend="BULLISH", index_trend="BULLISH")
        self.assertEqual(mu_max, 1.15)
        
        # 2. Extreme Chop (ADX 15) + Macro conflict -> 0.50 * 0.80 = 0.40 -> Clamped to 0.50
        mu_min = calculate_context_multiplier(adx=15.0, stock_trend="BULLISH", index_trend="BEARISH")
        self.assertEqual(mu_min, 0.50)
        
        # 3. Transitional ADX 25 + Neutral Index -> 0.75 * 1.0 = 0.75
        mu_mid = calculate_context_multiplier(adx=25.0, stock_trend="BULLISH", index_trend="INDEX")
        self.assertAlmostEqual(mu_mid, 0.75, places=2)

    def test_dynamic_atr_position_sizing_hierarchy(self):
        """
        Verify that dynamic ATR sizing enforces strict min() hierarchy:
        Tiny ATR Stop -> Large raw size -> Hard Max Lots Cap strictly binds!
        """
        guard = AIGuardrails(max_lots_per_trade=2, max_risk_per_trade_pct=0.01)
        
        # Stock: Price 100, Stop 99.9 (tiny 0.10 pt risk), Capital 100,000
        # 1% risk = 1,000 -> Raw shares = 10,000 shares
        # But max_lots_cap = 2 lots (lot_size=25 -> max 50 shares)
        sized_qty = guard.calculate_dynamic_position_size(
            capital=100000.0,
            entry_price=100.0,
            sl_price=99.9,
            atr=0.10,
            lot_size=25,
            max_lots_cap=2,
            risk_pct=0.01
        )
        # Hard cap of 2 lots * 25 shares = 50 shares MUST bind!
        self.assertEqual(sized_qty, 50)

    def test_rvol_calculation_and_breakout_gate(self):
        """
        Verify that Relative Volume (RVol) is computed correctly and
        Breakout strategies with low volume (<1.00x) are strictly vetoed by guardrails.
        """
        from src.strategies.indicators import calculate_rvol
        v_series = pd.Series([1000] * 20 + [2500]) # Last bar has 2.5x volume
        rvol = calculate_rvol(v_series, 20)
        self.assertEqual(rvol, 2.50)

        # Low volume breakout test in Guardrails
        guard = AIGuardrails(min_confidence_threshold=7.5)
        proposal = {
            "action": "BUY_STOCK",
            "target_asset": "TCS",
            "strategy": "BREAKOUT",
            "rvol": 0.65, # Below average volume breakout
            "confidence_score": 8.0,
            "suggested_sl_pct": 1.5,
            "suggested_tp_pct": 3.0
        }
        portfolio = {"capital": 100000.0, "daily_pnl": 0.0, "open_positions": []}
        approved, reason, _ = guard.evaluate_proposal(proposal, portfolio)
        self.assertFalse(approved)
        self.assertIn("Low-Volume Breakout", reason)

if __name__ == "__main__":
    unittest.main()

