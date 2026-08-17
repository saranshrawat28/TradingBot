"""
Automated Test Suite for Orthogonal Bucket Capping and ADX Regime Detection.
"""

import unittest
import pandas as pd
import numpy as np
from src.strategies.indicators import calculate_adx, add_all_indicators
from src.engine.stock_advisor import StockAdvisor

class TestOrthogonalScorerSuite(unittest.TestCase):

    def _create_mock_df(self, n_bars: int = 60, trend: str = "up") -> pd.DataFrame:
        np.random.seed(42)
        if trend == "up":
            closes = 100.0 + np.cumsum(np.random.uniform(0.1, 0.8, n_bars))
        elif trend == "down":
            closes = 200.0 - np.cumsum(np.random.uniform(0.1, 0.8, n_bars))
        else: # chop / sideways
            closes = 150.0 + np.sin(np.linspace(0, 20, n_bars)) * 0.4
            highs = closes + 0.3
            lows = closes - 0.3
            volumes = np.random.randint(10000, 50000, n_bars)
            return pd.DataFrame({"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": volumes})

        highs = closes + np.random.uniform(0.2, 1.0, n_bars)
        lows = closes - np.random.uniform(0.2, 1.0, n_bars)
        volumes = np.random.randint(10000, 50000, n_bars)
        
        return pd.DataFrame({
            "Open": closes,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes
        })

    def test_adx_regime_classification(self):
        """Verify ADX detects trending vs range-bound regimes."""
        # Strong uptrend df
        df_up = self._create_mock_df(60, "up")
        adx, pdi, mdi = calculate_adx(df_up["High"], df_up["Low"], df_up["Close"], 14)
        self.assertGreater(adx.iloc[-1], 25.0)
        self.assertGreater(pdi.iloc[-1], mdi.iloc[-1]) # Bullish DMI

        # Range-bound chop df
        df_chop = self._create_mock_df(60, "chop")
        adx_c, _, _ = calculate_adx(df_chop["High"], df_chop["Low"], df_chop["Close"], 14)
        self.assertLess(adx_c.iloc[-1], 25.0)

    def test_orthogonal_bucket_capping(self):
        """Verify that individual buckets never exceed their defined caps."""
        df = self._create_mock_df(80, "up")
        eval_res = StockAdvisor.evaluate_df_slice(df, "RELIANCE.NS")
        
        buckets = eval_res["buckets"]
        self.assertLessEqual(buckets["trend"], 2.5)
        self.assertLessEqual(buckets["momentum"], 2.0)
        self.assertLessEqual(buckets["volatility"], 1.5)
        self.assertLessEqual(buckets["volume_flow"], 1.5)

    def test_dynamic_atr_targets_computed(self):
        """Verify Stop-Loss and Targets dynamically scale with instrument ATR."""
        df = self._create_mock_df(60, "up")
        eval_res = StockAdvisor.evaluate_df_slice(df, "RELIANCE.NS")
        
        curr_p = eval_res["current_price"]
        atr = eval_res["metrics"]["atr"]
        levels = eval_res["levels"]
        
        expected_sl = round(curr_p - (1.2 * atr), 2)
        expected_t1 = round(curr_p + (1.8 * atr), 2)
        expected_t2 = round(curr_p + (3.0 * atr), 2)
        
        self.assertEqual(levels["stop_loss"], expected_sl)
        self.assertEqual(levels["target_1"], expected_t1)
        self.assertEqual(levels["target_2"], expected_t2)

if __name__ == "__main__":
    unittest.main()
