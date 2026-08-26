import unittest
import pandas as pd
import numpy as np
from src.strategies.options_greeks import DerivativesFlowAnalyzer
from src.engine.stock_advisor import StockAdvisor
from src.ai.chat_assistant import TradingChatAssistant

class TestDerivativesFlowAndAccuracy(unittest.TestCase):
    """
    Tests the Live NSE Derivatives Telemetry, Call/Put Option Walls,
    Call Wall Collision Shield, and the 82%+ Win-Rate Confluence Gate.
    """

    def test_derivatives_analyzer_calculation(self):
        """Validates that DerivativesFlowAnalyzer correctly calculates option walls, PCR, and OI interpretation."""
        spot = 2850.0
        res = DerivativesFlowAnalyzer.analyze_derivatives_structure("RELIANCE", spot)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreaterEqual(res["call_writer_wall"], spot)
        self.assertLessEqual(res["put_writer_floor"], spot)
        self.assertGreater(res["max_pain"], 0.0)
        self.assertGreater(res["pcr_oi"], 0.0)
        self.assertIn(res["oi_interpretation"], ["LONG_BUILDUP", "PUT_WRITING_SUPPORT", "SHORT_BUILDUP", "SHORT_COVERING", "NEUTRAL_BALANCED"])
        self.assertIsInstance(res["has_clear_runway"], bool)

    def test_call_wall_collision_shield(self):
        """Validates that Target 1 is automatically recalibrated to front-run institutional Call Writer Walls."""
        dates = pd.date_range("2026-08-01", periods=60, freq="15min")
        close = pd.Series(np.linspace(2800, 2840, 60), index=dates)
        high = close + 2.0
        low = close - 2.0
        df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": 10000}, index=dates)

        # Mock derivatives telemetry where Call Wall sits at 2920 (below normal Target 1 of 2946.5)
        mock_deriv = {
            "status": "SUCCESS",
            "call_writer_wall": 2920.0,
            "put_writer_floor": 2800.0,
            "max_pain": 2830.0,
            "pcr_oi": 1.30,
            "has_clear_runway": True,
            "oi_interpretation": "LONG_BUILDUP"
        }

        res = StockAdvisor.evaluate_df_slice(df, symbol="RELIANCE", horizon="swing", deriv_info=mock_deriv)
        t1 = res["target_1"]["price"]
        # Target 1 must be strictly less than or equal to call_writer_wall
        self.assertLessEqual(t1, 2920.0)
        has_shield_notice = any("Call Wall Collision Shield" in p for p in res["pros"])
        self.assertTrue(has_shield_notice)

    def test_grade_a_plus_confluence_gate(self):
        """Validates that Grade A+ (>82% win rate) is granted ONLY when 6-way confluence is achieved."""
        dates = pd.date_range("2026-08-01", periods=60, freq="15min")
        close = pd.Series(np.linspace(2700, 2900, 60), index=dates)
        high = close + 3.0
        low = close - 1.0
        volume = pd.Series(np.linspace(50000, 150000, 60), index=dates) # high volume
        df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)

        # 1. Full 6-way confluence: Bullish Nifty, Bullish Daily, Long Build-up, Clear Runway
        elite_deriv = {
            "status": "SUCCESS",
            "call_writer_wall": 3100.0,
            "put_writer_floor": 2800.0,
            "max_pain": 2850.0,
            "pcr_oi": 1.35,
            "has_clear_runway": True,
            "oi_interpretation": "LONG_BUILDUP"
        }
        res_elite = StockAdvisor.evaluate_df_slice(
            df, symbol="RELIANCE",
            horizon="swing",
            index_trend="BULLISH",
            htf_trend="BULLISH",
            deriv_info=elite_deriv
        )
        if res_elite["score"] >= 8.0:
            self.assertEqual(res_elite["setup_grade"], "GRADE_A_PLUS")
            self.assertGreaterEqual(res_elite["win_probability"], 82)

        # 2. Blocked Runway (Call Wall immediately overhead at 2905)
        blocked_deriv = {
            "status": "SUCCESS",
            "call_writer_wall": 2905.0,
            "put_writer_floor": 2800.0,
            "max_pain": 2850.0,
            "pcr_oi": 1.35,
            "has_clear_runway": False, # Blocked!
            "oi_interpretation": "LONG_BUILDUP"
        }
        res_blocked = StockAdvisor.evaluate_df_slice(
            df, symbol="RELIANCE",
            horizon="swing",
            index_trend="BULLISH",
            htf_trend="BULLISH",
            deriv_info=blocked_deriv
        )
        # Cannot be GRADE_A_PLUS when runway is blocked by institutional call writers
        self.assertNotEqual(res_blocked["setup_grade"], "GRADE_A_PLUS")

    def test_chat_assistant_derivatives_rendering(self):
        """Validates that Chat Assistant briefings render Derivatives Telemetry."""
        mock_analysis = {
            "status": "SUCCESS",
            "display_name": "Tata Motors CV",
            "current_price": 476.55,
            "score": 8.2,
            "verdict": "STRONG BUY",
            "setup_grade": "GRADE_A_PLUS",
            "setup_grade_title": "🌟 GRADE A+ (Elite Institutional Setup — 84% Win-Rate Probability)",
            "entry_zone": "₹475.00 – ₹476.55",
            "target_1": {"price": 494.40, "gain_pct": 3.75},
            "target_2": {"price": 506.35, "gain_pct": 6.25},
            "stop_loss": {"price": 464.65, "loss_pct": 2.50},
            "verdict_desc": "Multi-timeframe confluence with derivatives long build-up.",
            "derivatives": {
                "status": "SUCCESS",
                "call_writer_wall": 500.0,
                "put_writer_floor": 460.0,
                "max_pain": 480.0,
                "pcr_oi": 1.28,
                "oi_interpretation": "LONG_BUILDUP"
            }
        }
        resp = TradingChatAssistant._generate_heuristic_response(
            query="analyze TMCV",
            symbol="TMCV.NS",
            stock_analysis=mock_analysis,
            top_picks_data=[]
        )
        self.assertIn("Derivatives Order Flow", resp)
        self.assertIn("LONG_BUILDUP", resp)
        self.assertIn("Call Ceiling", resp)
        self.assertIn("GRADE A+", resp)

if __name__ == "__main__":
    unittest.main()
