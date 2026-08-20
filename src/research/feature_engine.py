"""
Systematic Feature Engineering and Quantitative Factor Store.
Transforms raw OHLCV and benchmark data into orthogonal, normalized predictive features
and forward target labels without lookahead bias.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

class FeatureEngine:
    """
    Computes institutional mathematical features and target labels from asset and benchmark price series.
    """
    
    FEATURE_COLUMNS = [
        "ret_1d", "ret_5d", "ret_20d", "ret_50d",
        "vol_parkinson_20", "vol_garman_klass_20", "atr_pct", "bollinger_width",
        "ema_spread_20_50", "ema_spread_50_200", "price_zscore_20",
        "rsi_norm", "volume_ratio_20", "benchmark_beta_20", "relative_strength_5d"
    ]
    
    @staticmethod
    def compute_all_features(
        df_asset: pd.DataFrame,
        df_benchmark: Optional[pd.DataFrame] = None,
        target_horizon: int = 5,
        target_hurdle_pct: float = 0.5
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Compute full normalized feature matrix and forward target labels.
        Returns:
            Tuple of (processed DataFrame with features and targets, list of feature column names)
        """
        if df_asset.empty or len(df_asset) < 60:
            return pd.DataFrame(), []
            
        df = df_asset.copy().sort_index()
        
        # Ensure standard column names
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns and col.lower() in df.columns:
                df[col] = df[col.lower()]
                
        # 1. Momentum & Return Factors
        df["ret_1d"] = df["Close"].pct_change(1) * 100.0
        df["ret_5d"] = df["Close"].pct_change(5) * 100.0
        df["ret_20d"] = df["Close"].pct_change(20) * 100.0
        df["ret_50d"] = df["Close"].pct_change(50) * 100.0
        
        # 2. Volatility Factors (Parkinson & Garman-Klass)
        log_hl = np.log(np.maximum(df["High"] / np.maximum(df["Low"], 1e-6), 1e-6))
        log_co = np.log(np.maximum(df["Close"] / np.maximum(df["Open"], 1e-6), 1e-6))
        
        # Parkinson High-Low Volatility (annualized)
        parkinson_var = (log_hl ** 2) / (4.0 * np.log(2.0))
        df["vol_parkinson_20"] = np.sqrt(parkinson_var.rolling(20).mean() * 252.0) * 100.0
        
        # Garman-Klass Volatility (annualized)
        gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        df["vol_garman_klass_20"] = np.sqrt(np.maximum(0.0, gk_var.rolling(20).mean() * 252.0)) * 100.0
        
        # ATR % Ratio
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Close"].shift(1)).abs()
        tr3 = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        df["atr_pct"] = (atr_14 / np.maximum(df["Close"], 1e-6)) * 100.0
        
        # Bollinger Bandwidth
        sma_20 = df["Close"].rolling(20).mean()
        std_20 = df["Close"].rolling(20).std()
        df["bollinger_width"] = ((2.0 * std_20 * 2.0) / np.maximum(sma_20, 1e-6)) * 100.0
        
        # 3. Moving Average Relationships & Z-Score
        ema_20 = df["Close"].ewm(span=20, adjust=False).mean()
        ema_50 = df["Close"].ewm(span=50, adjust=False).mean()
        ema_200 = df["Close"].ewm(span=200, adjust=False).mean()
        
        df["ema_spread_20_50"] = ((ema_20 - ema_50) / np.maximum(ema_50, 1e-6)) * 100.0
        df["ema_spread_50_200"] = ((ema_50 - ema_200) / np.maximum(ema_200, 1e-6)) * 100.0
        df["price_zscore_20"] = (df["Close"] - sma_20) / np.maximum(std_20, 1e-6)
        
        # 4. Normalized RSI (Scale from -1.0 to +1.0)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / np.maximum(loss, 1e-6)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        df["rsi_norm"] = (rsi - 50.0) / 50.0  # -1.0 is oversold 0, +1.0 is overbought 100
        
        # 5. Volume Momentum
        vol = df["Volume"] if "Volume" in df.columns else pd.Series(100000, index=df.index)
        vol_sma_20 = vol.rolling(20).mean()
        df["volume_ratio_20"] = vol / np.maximum(vol_sma_20, 1e-6)
        
        # 6. Benchmark Beta & Relative Strength
        if df_benchmark is not None and not df_benchmark.empty and len(df_benchmark) > 30:
            bench = df_benchmark["Close"].reindex(df.index).ffill().bfill()
            bench_ret = bench.pct_change(1)
            asset_ret = df["Close"].pct_change(1)
            
            # Rolling 20-day Covariance & Variance for Beta
            cov_20 = asset_ret.rolling(20).cov(bench_ret)
            var_20 = bench_ret.rolling(20).var()
            df["benchmark_beta_20"] = (cov_20 / np.maximum(var_20, 1e-6)).clip(-3.0, 3.0).fillna(1.0)
            
            # Relative Return vs Benchmark over 5 days
            bench_ret_5d = bench.pct_change(5) * 100.0
            df["relative_strength_5d"] = df["ret_5d"] - bench_ret_5d
        else:
            df["benchmark_beta_20"] = 1.0
            df["relative_strength_5d"] = df["ret_5d"]
            
        # 7. Forward Target Generation (No Lookahead Leakage during training)
        df["forward_ret_h"] = (df["Close"].shift(-target_horizon) - df["Close"]) / df["Close"] * 100.0
        df["target_outperform"] = (df["forward_ret_h"] > target_hurdle_pct).astype(int)
        
        # Clean infinite values and initial lookback NaNs
        feature_cols = [col for col in FeatureEngine.FEATURE_COLUMNS if col in df.columns]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        
        # Drop rows where rolling lookbacks are uninitialized (e.g. first 50 rows)
        clean_df = df.iloc[50:].copy()
        
        return clean_df, feature_cols
