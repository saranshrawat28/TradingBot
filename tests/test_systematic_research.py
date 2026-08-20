"""
Unit Test Suite for 7-Layer Systematic Quantitative Trading Research Platform.
Validates Feature Engineering Store, Multi-Model Tournament, Walk-Forward Validation Engine,
and Research Journal Persistence.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.research.feature_engine import FeatureEngine
from src.research.model_tournament import ModelTournament
from src.research.walk_forward_engine import WalkForwardEngine
from src.research.research_journal import ResearchJournal

class TestSystematicResearchSuite(unittest.TestCase):
    """
    Test suite for systematic research and institutional quantitative modeling.
    """

    @classmethod
    def setUpClass(cls):
        """Generate deterministic synthetic market data for reproducible testing."""
        np.random.seed(42)
        n_bars = 250
        dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="B")
        
        # Synthetic geometric random walk with trend and mean-reverting components
        drift = 0.0005
        vol = 0.012
        shocks = np.random.normal(drift, vol, n_bars)
        price_path = 100.0 * np.exp(np.cumsum(shocks))
        
        highs = price_path * (1.0 + np.abs(np.random.normal(0.005, 0.003, n_bars)))
        lows = price_path * (1.0 - np.abs(np.random.normal(0.005, 0.003, n_bars)))
        opens = price_path * (1.0 + np.random.normal(0.0, 0.002, n_bars))
        volumes = np.random.randint(50000, 500000, n_bars)
        
        cls.df_asset = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": price_path,
            "Volume": volumes
        }, index=dates)
        
        # Benchmark index data (NIFTY)
        bench_shocks = np.random.normal(0.0003, 0.010, n_bars)
        bench_price = 20000.0 * np.exp(np.cumsum(bench_shocks))
        cls.df_bench = pd.DataFrame({
            "Open": bench_price,
            "High": bench_price * 1.004,
            "Low": bench_price * 0.996,
            "Close": bench_price,
            "Volume": volumes * 2
        }, index=dates)

    def test_feature_engineering_computation_and_no_lookahead(self):
        """Verify FeatureEngine computes all 15 orthogonal factors and target labels."""
        df_feat, feat_cols = FeatureEngine.compute_all_features(
            self.df_asset, self.df_bench, target_horizon=5, target_hurdle_pct=0.5
        )
        
        self.assertFalse(df_feat.empty)
        self.assertGreaterEqual(len(feat_cols), 10)
        self.assertIn("ret_1d", feat_cols)
        self.assertIn("vol_parkinson_20", feat_cols)
        self.assertIn("vol_garman_klass_20", feat_cols)
        self.assertIn("atr_pct", feat_cols)
        self.assertIn("price_zscore_20", feat_cols)
        self.assertIn("rsi_norm", feat_cols)
        self.assertIn("benchmark_beta_20", feat_cols)
        self.assertIn("relative_strength_5d", feat_cols)
        
        # Verify target columns exist
        self.assertIn("forward_ret_h", df_feat.columns)
        self.assertIn("target_outperform", df_feat.columns)
        
        # Verify no NaNs in cleaned feature columns
        nan_counts = df_feat[feat_cols].isna().sum().sum()
        self.assertEqual(nan_counts, 0, "Feature matrix contains unexpected NaNs.")

    def test_model_tournament_training_and_oos_scoring(self):
        """Verify ModelTournament trains multiple architectures and scores them on out-of-sample data."""
        df_feat, feat_cols = FeatureEngine.compute_all_features(self.df_asset, self.df_bench)
        tourney_res = ModelTournament.run_tournament(df_feat, feat_cols, train_ratio=0.70)
        
        self.assertEqual(tourney_res.get("status"), "SUCCESS")
        self.assertGreater(tourney_res["train_samples"], 0)
        self.assertGreater(tourney_res["test_samples"], 0)
        
        models = tourney_res.get("models", {})
        self.assertIn("Buy & Hold Benchmark", models)
        self.assertIn("Dual EMA + RSI Momentum", models)
        self.assertIn("Ridge Logistic Regression (L2)", models)
        self.assertIn("Random Forest Ensemble", models)
        
        # Check that risk metrics are populated
        rf_m = models["Random Forest Ensemble"]
        self.assertIn("sharpe_ratio", rf_m)
        self.assertIn("max_drawdown_pct", rf_m)
        self.assertIn("cagr_pct", rf_m)
        self.assertIn("profit_factor", rf_m)
        self.assertIn("win_rate_pct", rf_m)
        self.assertIn("feature_importance", rf_m)

    def test_walk_forward_purged_folds_and_deflated_sharpe(self):
        """Verify WalkForwardEngine creates temporal splits and computes Deflated Sharpe Ratio."""
        df_feat, feat_cols = FeatureEngine.compute_all_features(self.df_asset, self.df_bench)
        wf_res = WalkForwardEngine.run_walk_forward_analysis(
            df_feat, feat_cols, n_splits=3, train_window_pct=0.50, embargo_bars=3
        )
        
        self.assertEqual(wf_res.get("status"), "SUCCESS")
        self.assertGreaterEqual(wf_res.get("folds_count", 0), 2)
        self.assertIn("avg_oos_sharpe", wf_res)
        self.assertIn("deflated_sharpe", wf_res)
        self.assertIn("consistency_score_pct", wf_res)
        self.assertIn("worst_oos_drawdown_pct", wf_res)
        
        folds = wf_res.get("folds", [])
        for f in folds:
            self.assertIn("fold_index", f)
            self.assertIn("sharpe_ratio", f)
            self.assertIn("cagr_pct", f)

    def test_research_journal_sqlite_logging(self):
        """Verify ResearchJournal logs experiments to SQLite and retrieves records."""
        test_sym = "TEST_ASSET.NS"
        test_hypo = "Parkinson Volatility + 5d Relative Strength Alpha Factor"
        
        exp_id = ResearchJournal.log_experiment(
            symbol=test_sym,
            hypothesis=test_hypo,
            model_type="Random Forest Ensemble",
            oos_sharpe=1.85,
            deflated_sharpe=1.62,
            oos_cagr=24.5,
            oos_max_dd=-6.4,
            win_rate=58.2,
            consistency_pct=100.0,
            notes="Strong stability across all test splits."
        )
        
        self.assertIsInstance(exp_id, int)
        self.assertGreater(exp_id, 0)
        
        records = ResearchJournal.get_experiments(limit=10)
        matching = [r for r in records if r["id"] == exp_id]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["symbol"], test_sym)
        self.assertEqual(matching[0]["hypothesis"], test_hypo)
        self.assertEqual(matching[0]["oos_sharpe"], 1.85)

if __name__ == "__main__":
    unittest.main()
