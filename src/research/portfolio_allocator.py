"""
Institutional Multi-Asset Portfolio Construction & Risk Allocation Engine.
Implements Hierarchical Risk Parity (HRP), Inverse Volatility Parity,
and Fractional Kelly Allocation with portfolio risk constraints using pure NumPy/Pandas.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

class PortfolioAllocator:
    """
    Constructs risk-optimal multi-asset portfolio allocations.
    Protects against single-stock concentration risk and matrix inversion singularities.
    """
    
    @staticmethod
    def compute_inverse_volatility_weights(returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Allocate capital inversely proportional to individual asset volatility (Risk Parity).
        w_i = (1 / sigma_i) / sum(1 / sigma_j)
        """
        if returns_df.empty:
            return {}
            
        vols = returns_df.std() * np.sqrt(252.0)
        vols = vols.replace(0.0, np.nan).fillna(0.01)
        
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()
        
        return {col: round(float(weights[col]), 4) for col in returns_df.columns}
        
    @staticmethod
    def compute_hrp_weights(returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (HRP - Marcos López de Prado standard).
        Uses hierarchical tree clustering on correlation distances to equalize risk across clusters.
        """
        if returns_df.empty or len(returns_df.columns) < 2:
            return {col: 1.0 / max(1, len(returns_df.columns)) for col in returns_df.columns}
            
        cov = returns_df.cov() * 252.0
        corr = returns_df.corr().fillna(0.0)
        
        # 1. Compute Distance Matrix: D_i,j = sqrt(0.5 * (1 - rho_i,j))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0.0, 1.0))
        
        # 2. Quasi-Diagonalization via Hierarchical Agglomerative Clustering
        cols = list(returns_df.columns)
        sorted_indices = PortfolioAllocator._quasi_diagonalize(dist)
        sorted_cols = [cols[i] for i in sorted_indices]
        
        # 3. Recursive Bisection for Cluster Weight Allocation
        sorted_cov = cov.loc[sorted_cols, sorted_cols]
        weights_series = PortfolioAllocator._recursive_bisection(sorted_cov, sorted_cols)
        
        return {col: round(float(weights_series.get(col, 0.0)), 4) for col in cols}

    @staticmethod
    def _quasi_diagonalize(dist_matrix: np.ndarray) -> List[int]:
        """Fast agglomerative hierarchical sorting of distance matrix."""
        n = dist_matrix.shape[0]
        if n <= 1:
            return list(range(n))
            
        # Greedy linkage ordering
        unvisited = set(range(n))
        first_node = 0
        ordered = [first_node]
        unvisited.remove(first_node)
        
        while unvisited:
            last = ordered[-1]
            next_node = min(unvisited, key=lambda x: dist_matrix[last, x])
            ordered.append(next_node)
            unvisited.remove(next_node)
            
        return ordered

    @staticmethod
    def _recursive_bisection(cov_df: pd.DataFrame, items: List[str]) -> pd.Series:
        """Recursively bisect clusters and allocate inverse cluster-variance weights."""
        weights = pd.Series(1.0, index=items)
        clusters = [items]
        
        while len(clusters) > 0:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]
                    
                    # Compute cluster variance for left and right partitions
                    cov_left = cov_df.loc[left, left]
                    cov_right = cov_df.loc[right, right]
                    
                    var_left = PortfolioAllocator._get_cluster_variance(cov_left)
                    var_right = PortfolioAllocator._get_cluster_variance(cov_right)
                    
                    # Compute split factor alpha = 1 - var_left / (var_left + var_right)
                    total_var = var_left + var_right
                    alpha = 1.0 - (var_left / max(1e-6, total_var))
                    alpha = np.clip(alpha, 0.05, 0.95)
                    
                    weights[left] *= alpha
                    weights[right] *= (1.0 - alpha)
                    
                    new_clusters.append(left)
                    new_clusters.append(right)
            clusters = new_clusters
            
        # Normalize sum to 1.0
        return weights / weights.sum()

    @staticmethod
    def _get_cluster_variance(cov_matrix: pd.DataFrame) -> float:
        """Compute variance of an inverse-volatility weighted cluster."""
        iv_weights = 1.0 / np.sqrt(np.diag(cov_matrix))
        iv_weights /= iv_weights.sum()
        return float(np.dot(np.dot(iv_weights, cov_matrix.values), iv_weights))

    @staticmethod
    def compute_fractional_kelly_weights(
        returns_df: pd.DataFrame,
        expected_excess_returns: Dict[str, float],
        fraction: float = 0.30
    ) -> Dict[str, float]:
        """
        Fractional Kelly Criterion: sizes weights proportional to expected alpha / variance.
        w_i = kappa * (mu_i / sigma_i^2)
        """
        if returns_df.empty:
            return {}
            
        vols = returns_df.std() * np.sqrt(252.0)
        variances = (vols ** 2).replace(0.0, np.nan).fillna(0.001)
        
        kelly_scores = {}
        for col in returns_df.columns:
            mu = expected_excess_returns.get(col, 10.0) / 100.0  # Default 10% expected excess
            var = variances[col]
            raw_kelly = fraction * (mu / max(0.001, var))
            kelly_scores[col] = max(0.0, raw_kelly)
            
        total = sum(kelly_scores.values())
        if total <= 0:
            return {col: 1.0 / len(returns_df.columns) for col in returns_df.columns}
            
        return {col: round(float(kelly_scores[col] / total), 4) for col in returns_df.columns}

    @staticmethod
    def apply_portfolio_constraints(
        weights: Dict[str, float],
        max_weight: float = 0.25,
        min_weight: float = 0.02
    ) -> Dict[str, float]:
        """
        Enforce institutional concentration constraints:
        Clamps individual asset weights to [min_weight, max_weight] and projects to simplex (sum = 1.0).
        """
        if not weights:
            return {}
            
        cols = list(weights.keys())
        w_arr = np.array([weights[c] for c in cols], dtype=float)
        
        # Iterative clipping & redistribution
        for _ in range(20):
            w_arr = np.clip(w_arr, min_weight, max_weight)
            diff = 1.0 - np.sum(w_arr)
            if abs(diff) < 1e-5:
                break
            # Distribute residual among unclipped weights
            unclipped_mask = (w_arr > min_weight + 1e-4) & (w_arr < max_weight - 1e-4)
            if np.sum(unclipped_mask) > 0:
                w_arr[unclipped_mask] += diff / np.sum(unclipped_mask)
            else:
                w_arr /= np.sum(w_arr)
                break
                
        w_arr = w_arr / np.sum(w_arr)
        return {cols[i]: round(float(w_arr[i]), 4) for i in range(len(cols))}

    @staticmethod
    def calculate_portfolio_telemetry(
        weights: Dict[str, float],
        returns_df: pd.DataFrame,
        risk_free_rate: float = 6.5
    ) -> Dict[str, float]:
        """
        Compute portfolio-level statistics: Expected Return, Volatility, Sharpe, and Diversification Ratio.
        """
        if returns_df.empty or not weights:
            return {"expected_return_pct": 0.0, "volatility_pct": 0.0, "sharpe_ratio": 0.0, "diversification_ratio": 1.0}
            
        cols = [c for c in returns_df.columns if c in weights]
        w_vec = np.array([weights[c] for c in cols])
        
        # Asset annualized returns and covariance
        mean_ret = returns_df[cols].mean() * 252.0 * 100.0
        cov_matrix = returns_df[cols].cov() * 252.0 * 10000.0  # in pct^2
        individual_vols = returns_df[cols].std() * np.sqrt(252.0) * 100.0
        
        port_return = float(np.dot(w_vec, mean_ret.values))
        port_variance = float(np.dot(w_vec, np.dot(cov_matrix.values, w_vec)))
        port_vol = float(np.sqrt(max(0.01, port_variance)))
        
        sharpe = round((port_return - risk_free_rate) / max(0.01, port_vol), 2)
        
        # Diversification Ratio = Weighted Individual Vols / Portfolio Vol
        weighted_vol_sum = float(np.dot(w_vec, individual_vols.values))
        div_ratio = round(weighted_vol_sum / max(0.01, port_vol), 2)
        
        return {
            "expected_return_pct": round(port_return, 2),
            "volatility_pct": round(port_vol, 2),
            "sharpe_ratio": sharpe,
            "diversification_ratio": div_ratio
        }
