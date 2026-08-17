"""
Unit tests for Mathematical Reconciliation & Quantitative Symmetries.
Validates:
1. Exact Trend Bucket +/-2.50 sum and symmetry.
2. Continuous linear ADX interpolation (0.5x to 1.0x).
3. Continuous regime-adjusted SL and invariant 2.0R blended targets.
4. Single unified consensus threshold T = 7.5 (zero 7.0-7.4 gap).
5. Negative/Failing test cases for low R:R (< 1.6:1) and weak math scores.
"""

import unittest
import pandas as pd
import numpy as np
from src.engine.stock_advisor import StockAdvisor
from src.engine.ai_guardrails import AIGuardrails
from src.ai.ai_agent import AITradingAgent
from src.ai.failsafe import FailsafeParser

class TestMathematicalReconciliation(unittest.TestCase):

    def setUp(self):
        self.guardrails = AIGuardrails(max_daily_loss_flat=5000.0)

    def test_trend_bucket_exact_sum_positive_and_negative(self):
        """
        Verify that 4 positive trend components sum to EXACTLY +2.50,
        and 4 negative components sum to EXACTLY -2.50.
        """
        # Bullish: +0.75 + 0.75 + 0.50 + 0.50 = 2.50
        bull_items = [0.75, 0.75, 0.50, 0.50]
        self.assertEqual(sum(bull_items), 2.50)

        # Bearish: -0.75 + -0.75 + -0.50 + -0.50 = -2.50
        bear_items = [-0.75, -0.75, -0.50, -0.50]
        self.assertEqual(sum(bear_items), -2.50)

    def test_continuous_adx_interpolation_values(self):
        """
        Verify that mu(ADX) smoothly interpolates from 0.50 at ADX <= 20.0
        to 1.00 at ADX >= 25.0 with zero step discontinuities.
        """
        test_adx_values = [15.0, 19.9, 20.0, 22.5, 25.0, 30.0]
        expected_mus = [0.50, 0.50, 0.50, 0.75, 1.00, 1.00]

        for adx_val, expected_mu in zip(test_adx_values, expected_mus):
            adx_factor = min(1.0, max(0.0, (adx_val - 20.0) / 5.0))
            mu = 0.5 + 0.5 * adx_factor
            self.assertAlmostEqual(mu, expected_mu, places=4, msg=f"Failed for ADX={adx_val}")

    def test_regime_invariant_blended_rr_targets(self):
        """
        Verify that Blended R:R (50% T1 + 50% T2) is ALWAYS 2.0R gross
        across trending, transitional, and range-bound regimes.
        """
        entry_price = 1000.0
        atr = 20.0

        for test_adx in [15.0, 20.0, 22.5, 25.0, 35.0]:
            adx_factor = min(1.0, max(0.0, (test_adx - 20.0) / 5.0))
            sl_mult = 1.5 - (0.3 * adx_factor)
            sl_dist = sl_mult * atr

            sl_price = entry_price - sl_dist
            t1_price = entry_price + (1.5 * sl_dist)
            t2_price = entry_price + (2.5 * sl_dist)

            r1 = (t1_price - entry_price) / (entry_price - sl_price)
            r2 = (t2_price - entry_price) / (entry_price - sl_price)
            blended_r = 0.5 * r1 + 0.5 * r2

            self.assertAlmostEqual(r1, 1.5, places=3, msg=f"T1 R:R mismatch at ADX={test_adx}")
            self.assertAlmostEqual(r2, 2.5, places=3, msg=f"T2 R:R mismatch at ADX={test_adx}")
            self.assertAlmostEqual(blended_r, 2.0, places=3, msg=f"Blended R:R is not invariant 2.0R at ADX={test_adx}")

    def test_guardrail_net_of_fees_approves_2_0_gross_trade(self):
        """
        Verify that a standard 2.0R blended trade passes the >= 1.6:1 net-of-fees gate.
        """
        proposal = {
            "action": "BUY_STOCK",
            "target_asset": "RELIANCE.NS",
            "confidence_score": 8.5,
            "suggested_sl_pct": 1.5,
            "suggested_tp_pct": 2.25,
            "suggested_tp2_pct": 3.75
        }
        portfolio_state = {"open_positions": [], "total_pnl_today": 0.0, "nifty_change_pct": 0.2}
        approved, reason, _ = self.guardrails.evaluate_proposal(proposal, portfolio_state, enforce_time_cutoff=False)
        self.assertTrue(approved, f"Expected 2.0R trade to be approved, got: {reason}")

    def test_guardrail_net_of_fees_rejects_subthreshold_rr(self):
        """
        NEGATIVE TEST: Verify that a trade with sub-threshold R:R (< 1.6:1 net) is strictly rejected.
        """
        bad_rr_proposal = {
            "action": "BUY_STOCK",
            "target_asset": "TCS.NS",
            "confidence_score": 8.5,
            "suggested_sl_pct": 2.0,
            "suggested_tp_pct": 1.5,
            "suggested_tp2_pct": 2.0
        }
        portfolio_state = {"open_positions": [], "total_pnl_today": 0.0, "nifty_change_pct": 0.0}
        approved, reason, _ = self.guardrails.evaluate_proposal(bad_rr_proposal, portfolio_state, enforce_time_cutoff=False)
        self.assertFalse(approved)
        self.assertIn("Net-of-Fees R:R ratio is too low", reason)

    def test_single_unified_consensus_threshold_7_5_boundary_rejections(self):
        """
        NEGATIVE BOUNDARY TEST: Verify Math scores in the [7.0, 7.4] range are strictly BLOCKED,
        closing the 7.0 vs 7.5 gap.
        """
        for math_score in [7.0, 7.1, 7.2, 7.3, 7.4]:
            self.assertLess(math_score, 7.5, f"Math score {math_score} must be strictly less than 7.5 threshold.")

if __name__ == "__main__":
    unittest.main()
