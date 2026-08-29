"""
Unit Tests for ApexTrade Paper Trading Accuracy Lab & Self-Diagnostic Engine.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date

from src.paper_lab.lab_config import LabConfig
from src.paper_lab.paper_db import PaperDB
from src.paper_lab.holiday_calendar import is_trading_day
from src.paper_lab.chronological_evaluator import ChronologicalEvaluator
from src.paper_lab.report_generator import ReportGenerator

class TestPaperLabSuite(unittest.TestCase):
    """Comprehensive test suite for Paper Trading Lab."""

    def setUp(self):
        PaperDB.init_db()

    def test_holiday_calendar(self):
        """Test trading day verifier on weekend vs weekday."""
        sat = datetime(2026, 8, 29) # Saturday
        sun = datetime(2026, 8, 30) # Sunday
        mon = datetime(2026, 8, 24) # Monday

        is_sat, r_sat = is_trading_day(sat)
        is_sun, r_sun = is_trading_day(sun)
        is_mon, r_mon = is_trading_day(mon)

        self.assertFalse(is_sat)
        self.assertIn("Saturday", r_sat)
        self.assertFalse(is_sun)
        self.assertIn("Sunday", r_sun)
        self.assertTrue(is_mon)

    def test_paper_db_idempotency(self):
        """Test saving picks and ensuring duplicate picks for the same date & symbol are skipped."""
        test_date = "2026-09-01"
        picks = [
            {
                "pick_date": test_date,
                "symbol": "TEST_STOCK_1.NS",
                "display_name": "Test Stock 1",
                "signal_time": "08:50:00",
                "signal_price": 1000.0,
                "target_1": 1030.0,
                "target_2": 1060.0,
                "stop_loss": 980.0,
                "allocated_capital": 20000.0,
                "quantity": 20,
                "advisor_score": 8.5,
                "score_breakdown": {"rsi": 55.0, "rvol": 1.5, "adx": 28.0, "vwap_sigma_dist": 0.15},
                "config_version": "v1.0.0",
                "status": "PENDING_OPEN"
            }
        ]

        count_1 = PaperDB.save_pending_picks(picks)
        # Attempt duplicate insert
        count_2 = PaperDB.save_pending_picks(picks)

        self.assertEqual(count_2, 0, "Duplicate pick should be ignored due to UNIQUE constraint.")

        # Update fill
        PaperDB.update_pick_fill(
            pick_date=test_date,
            symbol="TEST_STOCK_1.NS",
            entry_price=1005.0,
            entry_time="09:15:00",
            target_1=1035.0,
            target_2=1065.0,
            stop_loss=985.0,
            quantity=19
        )

        saved = PaperDB.get_picks_by_date(test_date)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["entry_price"], 1005.0)
        self.assertEqual(saved[0]["status"], "ACTIVE")

    def test_chronological_evaluator_target_first(self):
        """
        Verify that if Target 1 is touched at 10:15 and Stop Loss is touched at 13:30,
        the outcome is correctly resolved as T1_HIT.
        """
        # Create synthetic 1-minute bars
        times = pd.date_range("2026-09-01 09:15", "2026-09-01 15:25", freq="1min")
        n = len(times)

        # Baseline price = 100.0
        # Target 1 = 103.0, Stop Loss = 98.0
        opens = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        closes = np.full(n, 100.0)

        # Spike to 103.5 (T1 Hit) at index 60 (10:15 AM)
        highs[60] = 103.5
        closes[60] = 103.2

        # Crash to 97.0 (SL Hit) later at index 250 (01:25 PM)
        lows[250] = 97.0
        closes[250] = 97.2

        df_synthetic = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes
        }, index=times)

        pick = {
            "pick_date": "2026-09-01",
            "symbol": "SYNTH_T1.NS",
            "entry_price": 100.0,
            "target_1": 103.0,
            "target_2": 106.0,
            "stop_loss": 98.0,
            "quantity": 100,
            "allocated_capital": 10000.0
        }

        outcome = ChronologicalEvaluator.evaluate_pick_candles(pick, df_synthetic)
        self.assertEqual(outcome["exit_type"], "T1_HIT", "T1 touched first should resolve as T1_HIT.")
        self.assertEqual(outcome["exit_price"], 103.0)
        self.assertEqual(outcome["pnl_rs"], 300.0)
        self.assertEqual(outcome["bars_held"], 61)

    def test_chronological_evaluator_stoploss_first(self):
        """
        Verify that if Stop Loss is touched at 09:45 before Target 1 is touched at 14:00,
        the outcome is correctly resolved as SL_HIT.
        """
        times = pd.date_range("2026-09-01 09:15", "2026-09-01 15:25", freq="1min")
        n = len(times)

        opens = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        closes = np.full(n, 100.0)

        # Drop to 97.5 (SL Hit) at index 30 (09:45 AM)
        lows[30] = 97.5
        closes[30] = 97.8

        # Spike to 104.0 (Target) later at index 280 (01:55 PM)
        highs[280] = 104.0

        df_synthetic = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes
        }, index=times)

        pick = {
            "pick_date": "2026-09-01",
            "symbol": "SYNTH_SL.NS",
            "entry_price": 100.0,
            "target_1": 103.0,
            "target_2": 106.0,
            "stop_loss": 98.0,
            "quantity": 100,
            "allocated_capital": 10000.0
        }

        outcome = ChronologicalEvaluator.evaluate_pick_candles(pick, df_synthetic)
        self.assertEqual(outcome["exit_type"], "SL_HIT", "SL touched first should resolve as SL_HIT.")
        self.assertEqual(outcome["exit_price"], 98.0)
        self.assertEqual(outcome["pnl_rs"], -200.0)

    def test_chronological_evaluator_eod_close(self):
        """
        Verify that if neither SL nor Targets are touched, position exits at EOD close.
        """
        times = pd.date_range("2026-09-01 09:15", "2026-09-01 15:25", freq="1min")
        n = len(times)

        opens = np.full(n, 100.0)
        highs = np.full(n, 101.5)
        lows = np.full(n, 99.0)
        closes = np.full(n, 101.0) # Final close 101.0

        df_synthetic = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes
        }, index=times)

        pick = {
            "pick_date": "2026-09-01",
            "symbol": "SYNTH_EOD.NS",
            "entry_price": 100.0,
            "target_1": 103.0,
            "target_2": 106.0,
            "stop_loss": 98.0,
            "quantity": 100,
            "allocated_capital": 10000.0
        }

        outcome = ChronologicalEvaluator.evaluate_pick_candles(pick, df_synthetic)
        self.assertEqual(outcome["exit_type"], "EOD_CLOSE")
        self.assertEqual(outcome["exit_price"], 101.0)
        self.assertEqual(outcome["pnl_rs"], 100.0)

    def test_report_generator_diagnostic(self):
        """Test report generation and signal failure diagnostics aggregation."""
        date_str = "2026-09-05"

        # Create 5 synthetic picks
        picks = []
        for i in range(5):
            sym = f"DIAG_STOCK_{i}.NS"
            rsi_val = 72.0 if i < 3 else 52.0  # 3/5 had RSI overbought
            rvol_val = 0.60 if i < 2 else 1.4  # 2/5 had weak volume

            picks.append({
                "pick_date": date_str,
                "symbol": sym,
                "display_name": f"Diag Stock {i}",
                "signal_time": "08:50:00",
                "signal_price": 500.0,
                "entry_time": "09:15:00",
                "entry_price": 500.0,
                "target_1": 515.0,
                "target_2": 530.0,
                "stop_loss": 490.0,
                "allocated_capital": 20000.0,
                "quantity": 40,
                "advisor_score": 7.8,
                "setup_grade": "GRADE A",
                "score_breakdown": {
                    "rsi": rsi_val,
                    "rvol": rvol_val,
                    "vwap_sigma_dist": 0.20,
                    "adx": 24.0
                },
                "top_signals": ["Momentum breakout"],
                "config_version": "v1.0.0",
                "status": "ACTIVE"
            })

        PaperDB.save_pending_picks(picks)

        # Create 3 losses and 2 wins
        for i in range(5):
            sym = f"DIAG_STOCK_{i}.NS"
            is_win = (i >= 3)
            PaperDB.save_outcome({
                "pick_date": date_str,
                "symbol": sym,
                "entry_price": 500.0,
                "exit_price": 515.0 if is_win else 490.0,
                "exit_time": "11:30:00",
                "exit_type": "T1_HIT" if is_win else "SL_HIT",
                "pnl_rs": 600.0 if is_win else -400.0,
                "pnl_pct": 3.0 if is_win else -2.0,
                "allocated_capital": 20000.0,
                "quantity": 40,
                "bars_held": 45,
                "resolution_method": "1M_CANDLE_REPLAY"
            })

        report = ReportGenerator.generate_report(days_lookback=7, end_date=date_str)
        self.assertIn("financial_summary", report)
        self.assertEqual(report["prediction_accuracy"]["total_picks"], 5)
        self.assertEqual(report["prediction_accuracy"]["winning_picks"], 2)
        self.assertEqual(report["prediction_accuracy"]["losing_picks"], 3)
        self.assertEqual(report["prediction_accuracy"]["win_rate_pct"], 40.0)

        # Signal Diagnostics: 3/3 losers had RSI > 65
        diag = report["signal_diagnostics"]
        self.assertEqual(diag["rsi_fail_count"], 3)
        self.assertEqual(diag["rsi_fail_pct"], 100.0)
        self.assertIsNotNone(report["sample_warning"])

if __name__ == "__main__":
    unittest.main()
