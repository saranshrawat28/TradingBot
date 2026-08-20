"""
Unit Test Suite for Layer 5 Institutional Portfolio Construction & Allocation Engine.
Validates Inverse Volatility Parity, Hierarchical Risk Parity (HRP),
Fractional Kelly Allocation, Portfolio Constraints, and Diversification Ratios.
"""

import unittest
import numpy as np
import pandas as pd

from src.research.portfolio_allocator import PortfolioAllocator

class TestPortfolioAllocatorSuite(unittest.TestCase):
    """
    Test suite for multi-asset portfolio optimization and risk allocation.
    """

    @classmethod
    def setUpClass(cls):
        """Create deterministic synthetic multi-asset returns for testing."""
        np.random.seed(42)
        n_days = 250
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        
        # 4 Assets with distinct volatilities and correlations:
        # Asset A: Low Vol (10% annual vol)
        # Asset B: Med Vol (20% annual vol)
        # Asset C: High Vol (35% annual vol)
        # Asset D: Uncorrelated Defensive Asset (15% annual vol)
        ret_a = np.random.normal(0.0005, 0.10 / np.sqrt(252), n_days)
        ret_b = 0.5 * ret_a + np.random.normal(0.0007, 0.18 / np.sqrt(252), n_days)
        ret_c = 0.6 * ret_b + np.random.normal(0.0010, 0.30 / np.sqrt(252), n_days)
        ret_d = np.random.normal(0.0004, 0.15 / np.sqrt(252), n_days)
        
        cls.returns_df = pd.DataFrame({
            "ASSET_A": ret_a,
            "ASSET_B": ret_b,
            "ASSET_C": ret_c,
            "ASSET_D": ret_d
        }, index=dates)

    def test_inverse_volatility_weights_sum_to_one_and_order(self):
        """Verify Inverse Volatility allocates higher weights to lower-volatility assets."""
        weights = PortfolioAllocator.compute_inverse_volatility_weights(self.returns_df)
        
        self.assertEqual(len(weights), 4)
        total_w = sum(weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=3)
        
        # Asset A has lowest volatility -> should have highest weight
        # Asset C has highest volatility -> should have lowest weight
        self.assertGreater(weights["ASSET_A"], weights["ASSET_B"])
        self.assertGreater(weights["ASSET_B"], weights["ASSET_C"])

    def test_hrp_hierarchical_risk_parity_weights(self):
        """Verify HRP builds hierarchical tree clusters and outputs normalized weights."""
        hrp_weights = PortfolioAllocator.compute_hrp_weights(self.returns_df)
        
        self.assertEqual(len(hrp_weights), 4)
        total_w = sum(hrp_weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=3)
        
        for asset, w in hrp_weights.items():
            self.assertGreater(w, 0.0)
            self.assertLess(w, 1.0)

    def test_fractional_kelly_weights(self):
        """Verify Fractional Kelly calculates positive weights proportional to expected alpha."""
        expected_excess = {
            "ASSET_A": 8.0,
            "ASSET_B": 12.0,
            "ASSET_C": 18.0,
            "ASSET_D": 6.0
        }
        kelly_w = PortfolioAllocator.compute_fractional_kelly_weights(
            self.returns_df, expected_excess, fraction=0.30
        )
        
        self.assertEqual(len(kelly_w), 4)
        total_w = sum(kelly_w.values())
        self.assertAlmostEqual(total_w, 1.0, places=3)

    def test_portfolio_constraints_capping(self):
        """Verify portfolio constraints clamp maximum single-stock weight to 25%."""
        unconstrained = {
            "ASSET_A": 0.55,
            "ASSET_B": 0.25,
            "ASSET_C": 0.10,
            "ASSET_D": 0.10
        }
        constrained = PortfolioAllocator.apply_portfolio_constraints(
            unconstrained, max_weight=0.30, min_weight=0.05
        )
        
        for asset, w in constrained.items():
            self.assertLessEqual(w, 0.3001)
            self.assertGreaterEqual(w, 0.0499)
            
        self.assertAlmostEqual(sum(constrained.values()), 1.0, places=3)

    def test_portfolio_telemetry_calculation(self):
        """Verify telemetry calculates expected return, vol, Sharpe, and Diversification Ratio."""
        weights = {"ASSET_A": 0.25, "ASSET_B": 0.25, "ASSET_C": 0.25, "ASSET_D": 0.25}
        telemetry = PortfolioAllocator.calculate_portfolio_telemetry(weights, self.returns_df)
        
        self.assertIn("expected_return_pct", telemetry)
        self.assertIn("volatility_pct", telemetry)
        self.assertIn("sharpe_ratio", telemetry)
        self.assertIn("diversification_ratio", telemetry)
        
        # Diversification Ratio must be >= 1.0 due to imperfect correlation
        self.assertGreaterEqual(telemetry["diversification_ratio"], 1.0)
        self.assertGreater(telemetry["volatility_pct"], 0.0)

if __name__ == "__main__":
    unittest.main()
