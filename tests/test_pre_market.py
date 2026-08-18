"""
Unit Test Suite for Pre-Market Opening Analyzer & Intraday Stock Suggester.
"""

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

from src.engine.pre_market_analyzer import PreMarketAnalyzer

class TestPreMarketAnalyzer(unittest.TestCase):

    @patch("src.engine.pre_market_analyzer.get_live_quote")
    def test_market_opening_sentiment_bullish_gap(self, mock_quote):
        """Verify bullish gap-up classification when NIFTY gap >= +0.40%."""
        mock_quote.side_effect = lambda sym: {
            "^NSEI": {"price": 24500.0, "previous_close": 24350.0, "change_pct": 0.62},
            "^NSEBANK": {"price": 51500.0, "previous_close": 51000.0, "change_pct": 0.98},
            "^INDIAVIX": {"price": 12.8}
        }.get(sym, {"price": 100.0})

        sentiment = PreMarketAnalyzer.get_market_opening_sentiment()
        self.assertEqual(sentiment["sentiment"], "BULLISH_GAP_UP")
        self.assertIn("BULLISH OPEN EXPECTED", sentiment["title"])
        self.assertGreaterEqual(sentiment["gap_pct"], 0.40)
        self.assertIn("nifty_price", sentiment)

    @patch("src.engine.pre_market_analyzer.get_live_quote")
    def test_market_opening_sentiment_bearish_gap(self, mock_quote):
        """Verify bearish gap-down classification when NIFTY gap <= -0.40%."""
        mock_quote.side_effect = lambda sym: {
            "^NSEI": {"price": 24100.0, "previous_close": 24350.0, "change_pct": -1.02},
            "^NSEBANK": {"price": 50500.0, "previous_close": 51000.0, "change_pct": -0.98},
            "^INDIAVIX": {"price": 15.2}
        }.get(sym, {"price": 100.0})

        sentiment = PreMarketAnalyzer.get_market_opening_sentiment()
        self.assertEqual(sentiment["sentiment"], "BEARISH_GAP_DOWN")
        self.assertIn("BEARISH OPEN EXPECTED", sentiment["title"])
        self.assertLessEqual(sentiment["gap_pct"], -0.40)

    @patch("src.engine.pre_market_analyzer.get_historical_data")
    @patch("src.engine.pre_market_analyzer.get_live_quote")
    def test_scan_pre_market_stocks_generates_recommendations(self, mock_quote, mock_hist):
        """Verify pre-market stock scanning outputs top picks with clear targets, SL, and reasons."""
        dates = pd.date_range("2026-01-01 09:15", periods=50, freq="5min")
        df_bull = pd.DataFrame({
            "Open": np.linspace(100, 115, 50),
            "High": np.linspace(101, 116, 50),
            "Low": np.linspace(99, 114, 50),
            "Close": np.linspace(100.5, 115.5, 50),
            "Volume": [10000] * 50
        }, index=dates)

        mock_hist.return_value = df_bull
        mock_quote.return_value = {"price": 115.5, "previous_close": 112.0, "change_pct": 3.12}

        scan_res = PreMarketAnalyzer.scan_pre_market_stocks(universe=["RELIANCE.NS", "TMCV.NS"], top_n=2)
        
        self.assertIn("top_picks", scan_res)
        self.assertIn("opening_sentiment", scan_res)
        self.assertEqual(len(scan_res["top_picks"]), 2)

        pick = scan_res["top_picks"][0]
        self.assertIn("display_name", pick)
        self.assertIn("current_price", pick)
        self.assertIn("action", pick)
        self.assertIn("target_1_price", pick)
        self.assertIn("stop_loss_price", pick)
        self.assertIn("reason", pick)
        self.assertGreater(pick["target_1_price"], pick["current_price"])
        self.assertLess(pick["stop_loss_price"], pick["current_price"])

if __name__ == "__main__":
    unittest.main()
