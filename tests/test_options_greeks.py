"""
Automated Test Suite for Institutional Options Greeks, Dual IV Solver,
Open Interest Max Pain, and Gamma-Aware Strike Selection.
"""

import unittest
import math
from src.strategies.options_greeks import BlackScholesEngine, OptionChainBuilder, SmartStrikeSelector
from src.engine.ai_guardrails import AIGuardrails
from src.brokers.paper_broker import PaperBroker
from src.utils.storage import reset_all_data

class TestOptionsGreeksEngine(unittest.TestCase):

    def setUp(self):
        reset_all_data(100000.0)

    def test_numerical_correctness_against_published_reference(self):
        """
        Validate exact mathematical outputs against textbook Black-Scholes benchmark:
        Parameters: S = 100, K = 100, T = 1.0 year, r = 5% (0.05), sigma = 20% (0.20)
        Published Standard Values:
        - Call Price: 10.4506
        - Put Price: 5.5735
        - Call Delta: 0.6368
        - Put Delta: -0.3632
        - Gamma: 0.0188
        - Vega (1% IV): 0.3752
        - Daily Theta (CE): -0.0176
        """
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        
        # 1. Price Verification
        ce_price = BlackScholesEngine.calculate_option_price(s, k, t, r, sigma, "CE")
        pe_price = BlackScholesEngine.calculate_option_price(s, k, t, r, sigma, "PE")
        self.assertAlmostEqual(ce_price, 10.4506, places=2)
        self.assertAlmostEqual(pe_price, 5.5735, places=2)

        # 2. Greeks Verification
        ce_greeks = BlackScholesEngine.calculate_greeks(s, k, t, r, sigma, "CE")
        pe_greeks = BlackScholesEngine.calculate_greeks(s, k, t, r, sigma, "PE")
        
        # Call & Put Delta
        self.assertAlmostEqual(ce_greeks["delta"], 0.6368, places=2)
        self.assertAlmostEqual(pe_greeks["delta"], -0.3632, places=2)
        self.assertAlmostEqual(ce_greeks["delta"] - pe_greeks["delta"], 1.0000, places=2) # Put-Call Delta parity

        # Gamma
        self.assertAlmostEqual(ce_greeks["gamma"], 0.0188, places=3)
        self.assertAlmostEqual(pe_greeks["gamma"], 0.0188, places=3) # Gamma is identical for CE and PE

        # Vega (1% IV change)
        self.assertAlmostEqual(ce_greeks["vega"], 0.38, places=1)

        # Theta (Daily Decay < 0)
        self.assertLess(ce_greeks["theta_daily"], 0.0)
        self.assertAlmostEqual(ce_greeks["theta_daily"], -0.02, places=1)

    def test_dual_stage_iv_solver_and_fallbacks(self):
        """
        Verify Newton-Raphson solver and Bounded Bisection fallback:
        1. Invert exact price (10.45) -> Recovers ~20.0% IV.
        2. Intrinsic bound violation -> Gracefully returns None without crashing.
        3. Extreme / Near-zero quote -> Solves cleanly via bisection or returns safe bound.
        """
        s, k, t, r = 100.0, 100.0, 1.0, 0.05

        # Happy Path
        solved_iv = BlackScholesEngine.calculate_implied_volatility(10.45, s, k, t, r, "CE")
        self.assertIsNotNone(solved_iv)
        self.assertAlmostEqual(solved_iv, 20.0, delta=0.5)

        # Intrinsic lower bound violation (Price = 1.0 when intrinsic is 10.0)
        stale_quote_iv = BlackScholesEngine.calculate_implied_volatility(1.0, 100.0, 90.0, 0.1, 0.05, "CE")
        self.assertIsNone(stale_quote_iv)

        # Far OTM option with tiny premium
        otm_iv = BlackScholesEngine.calculate_implied_volatility(0.20, 100.0, 130.0, 0.1, 0.05, "CE")
        # Should either converge or return a bounded float without throwing an exception
        if otm_iv is not None:
            self.assertTrue(1.0 <= otm_iv <= 300.0)

    def test_open_interest_max_pain_and_pcr_arithmetic(self):
        """
        Verify Max Pain minimizes total option buyer payout and PCR arithmetic.
        """
        strikes_mock = [
            {"strike": 24400, "ce_oi": 5000, "pe_oi": 15000},
            {"strike": 24500, "ce_oi": 20000, "pe_oi": 20000}, # Heavy congestion
            {"strike": 24600, "ce_oi": 18000, "pe_oi": 6000},
        ]

        # Max Pain should align with peak congestion (24500)
        max_pain = OptionChainBuilder.calculate_max_pain(strikes_mock)
        self.assertEqual(max_pain, 24500.0)

        # PCR = Total PE OI (41,000) / Total CE OI (43,000) = 0.95
        pcr_res = OptionChainBuilder.calculate_pcr(strikes_mock)
        self.assertAlmostEqual(pcr_res["pcr_oi"], 0.95, places=2)

    def test_smart_strike_selector_gamma_awareness_on_0dte(self):
        """
        Verify that SmartStrikeSelector shifts from ATM to ITM1 on 0DTE (same-day expiry)
        to protect against extreme gamma spikes.
        """
        # 1. Normal trading day (DTE = 3.0 days) -> Selects ATM (Delta ~ 0.50)
        normal_sel = SmartStrikeSelector.select_optimal_strike("NIFTY", 24500.0, "BUY_CALL", dte_days=3.0, preference="ATM")
        self.assertEqual(normal_sel["strike"], 24500)
        self.assertFalse(normal_sel["is_0dte_adjusted"])

        # 2. Expiry Day (0DTE, DTE = 0.2 days) -> Automatically shifts to ITM1 (24450 CE)
        exp_sel = SmartStrikeSelector.select_optimal_strike("NIFTY", 24500.0, "BUY_CALL", dte_days=0.2, preference="ATM")
        self.assertEqual(exp_sel["strike"], 24450) # ITM 1 step lower for Call
        self.assertTrue(exp_sel["is_0dte_adjusted"])
        self.assertGreaterEqual(abs(exp_sel["target_delta"]), 0.60)

    def test_guardrail_routed_1_click_execution(self):
        """
        Verify that 1-click execution from Option Chain routes strictly through AIGuardrails.
        """
        guard = AIGuardrails(min_confidence_threshold=7.5)
        portfolio = {"open_positions": [], "realized_pnl": 0.0, "capital": 100000.0}

        # Proposal with low confidence (e.g. 6.0) -> MUST be blocked by guardrails
        proposal = {
            "symbol": "NIFTY 24500 CE",
            "action": "BUY_CALL",
            "confidence": 6.0,
            "entry_price": 120.0,
            "sl": 95.0,
            "target_1": 150.0,
            "target_2": 180.0
        }

        approved, reason, _ = guard.evaluate_proposal(proposal, portfolio, enforce_time_cutoff=False)
        self.assertFalse(approved)
        self.assertIn("safety threshold", reason.lower())

if __name__ == "__main__":
    unittest.main()
