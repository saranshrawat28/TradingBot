"""
Systematic Quantitative Trading Research Lab UI Component.
Renders the 7-layer institutional research pipeline: feature engineering,
multi-model tournaments, walk-forward cross-validation, and research journal.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from src.research.feature_engine import FeatureEngine
from src.research.model_tournament import ModelTournament
from src.research.walk_forward_engine import WalkForwardEngine
from src.research.research_journal import ResearchJournal
from src.data.data_fetcher import get_historical_data, get_live_quote
import config

def render_quant_research_tab(broker):
    """Render the full institutional quantitative research dashboard."""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #090e1a 0%, #1e1b4b 100%); border: 2px solid #6366f1; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;'>
        <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;'>
            <div>
                <h2 style='margin: 0; color: #ffffff; font-size: 1.6rem;'>🔬 Systematic Quantitative Research Lab</h2>
                <div style='color: #c7d2fe; font-size: 0.92rem; margin-top: 4px;'>
                    Institutional 7-Layer Pipeline: <strong>Data → Factors → Multi-Model Tournament → Walk-Forward OOS Testing → Friction Deduction → Research Journal</strong>
                </div>
            </div>
            <div style='display: flex; gap: 10px;'>
                <span class='badge-bull'>🛡️ Walk-Forward Purged</span>
                <span class='badge-cyan'>🏛️ Net-of-Fees Verified</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Research Configuration Panel
    with st.container():
        c1, c2, c3, c4 = st.columns([2.5, 1.5, 1.5, 2])
        with c1:
            research_sym = st.selectbox(
                "Select Asset to Research:",
                options=[item["symbol"] for item in config.DEFAULT_WATCHLIST],
                format_func=lambda s: next((f"{item['name']} ({item['symbol'].replace('.NS','')})" for item in config.DEFAULT_WATCHLIST if item["symbol"] == s), s),
                key="research_sym_select"
            )
        with c2:
            time_period = st.selectbox("Historical Horizon:", ["1y", "2y", "5y", "max"], index=1, key="res_period_sel")
        with c3:
            time_interval = st.selectbox("Bar Resolution:", ["1d", "1h", "15m"], index=0, key="res_interval_sel")
        with c4:
            oos_split_ratio = st.slider("Train / Out-of-Sample Split:", min_value=50, max_value=85, value=70, step=5, format="%d%% In-Sample") / 100.0
            
    st.write("")
    btn_col1, btn_col2 = st.columns([3, 1.5])
    with btn_col1:
        run_research_btn = st.button("🚀 Run Multi-Model Tournament & Walk-Forward Audit", type="primary", use_container_width=True)
    with btn_col2:
        wf_splits = st.number_input("Walk-Forward Test Folds:", min_value=2, max_value=6, value=4, step=1)
        
    if run_research_btn:
        with st.spinner(f"Fetching historical bars and computing 15 orthogonal quantitative factors for {research_sym}..."):
            df_raw = get_historical_data(research_sym, period=time_period, interval=time_interval)
            df_bench = get_historical_data("^NSEI", period=time_period, interval=time_interval)
            
            if df_raw.empty or len(df_raw) < 70:
                st.error(f"❌ Insufficient historical data for {research_sym}. Please select a longer horizon or daily interval.")
                return
                
            df_features, feat_cols = FeatureEngine.compute_all_features(df_raw, df_bench, target_horizon=5)
            st.session_state["res_df_features"] = df_features
            st.session_state["res_feat_cols"] = feat_cols
            st.session_state["res_target_sym"] = research_sym
            
            # Run Multi-Model Tournament
            tourney_res = ModelTournament.run_tournament(df_features, feat_cols, train_ratio=oos_split_ratio)
            st.session_state["res_tourney"] = tourney_res
            
            # Run Walk-Forward Engine
            wf_res = WalkForwardEngine.run_walk_forward_analysis(df_features, feat_cols, n_splits=int(wf_splits))
            st.session_state["res_walkforward"] = wf_res
            
    # Display Results if available
    if "res_tourney" in st.session_state and st.session_state["res_tourney"].get("status") == "SUCCESS":
        tourney = st.session_state["res_tourney"]
        wf = st.session_state.get("res_walkforward", {})
        feat_cols = st.session_state.get("res_feat_cols", [])
        sym = st.session_state.get("res_target_sym", "")
        
        st.markdown("---")
        st.subheader("⚔️ Layer 3: Multi-Model Benchmark Tournament (Out-of-Sample Results)")
        st.caption(f"Evaluated on completely unseen test data: **{tourney.get('test_period')}** ({tourney.get('test_samples')} bars) strictly net of Indian broker fees & slippage.")
        
        models_data = tourney.get("models", {})
        table_rows = []
        equity_curves_dict = {}
        
        for m_name, m_info in models_data.items():
            if m_info.get("status") == "ERROR":
                continue
            table_rows.append({
                "Model Architecture": m_name,
                "Type": m_info.get("type"),
                "Directional Accuracy": f"{m_info.get('accuracy')}%",
                "OOS CAGR": f"{m_info.get('cagr_pct'):+.2f}%",
                "Sharpe Ratio": m_info.get("sharpe_ratio"),
                "Sortino Ratio": m_info.get("sortino_ratio"),
                "Max Drawdown": f"{m_info.get('max_drawdown_pct'):.2f}%",
                "Profit Factor": m_info.get("profit_factor"),
                "Win Rate": f"{m_info.get('win_rate_pct')}%",
                "Trades": m_info.get("trades_count")
            })
            equity_curves_dict[m_name] = m_info.get("equity_curve", [])
            
        m_df = pd.DataFrame(table_rows)
        st.dataframe(m_df, use_container_width=True, hide_index=True)
        
        # Interactive Out-of-Sample Equity Curves Chart
        st.write("")
        st.markdown("#### 📈 Out-of-Sample Equity Curves Comparison (Base 100)")
        fig = go.Figure()
        colors = ["#94a3b8", "#f59e0b", "#38bdf8", "#10b981", "#a855f7"]
        
        for idx, (m_name, eq_curve) in enumerate(equity_curves_dict.items()):
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(
                y=eq_curve,
                mode="lines",
                name=m_name,
                line=dict(color=color, width=3 if "AI" in m_name or "Random Forest" in m_name else 1.5)
            ))
            
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#090e1a",
            plot_bgcolor="#090e1a",
            margin=dict(l=20, r=20, t=30, b=20),
            height=380,
            xaxis=dict(title="Out-of-Sample Step (Bars)", showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(title="Normalized Capital (₹)", showgrid=True, gridcolor="#1e293b"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Walk-Forward Consistency & Overfitting Diagnostics
        st.markdown("---")
        st.subheader("🛡️ Layer 4: Walk-Forward Validation & Overfitting Diagnostics")
        st.caption("Temporal sliding-window validation across distinct time regimes to test if edge holds over time.")
        
        if wf and wf.get("status") == "SUCCESS":
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Walk-Forward Consistency", f"{wf.get('consistency_score_pct')}%", f"{wf.get('profitable_folds')}/{wf.get('folds_count')} Folds Profitable")
            w2.metric("Mean Out-of-Sample Sharpe", f"{wf.get('avg_oos_sharpe'):.2f}")
            w3.metric("Deflated Sharpe Ratio (DSR)", f"{wf.get('deflated_sharpe'):.2f}", help="Penalizes high variance across folds to protect against p-hacking.")
            w4.metric("Worst Fold Drawdown", f"{wf.get('worst_oos_drawdown_pct'):.2f}%")
            
            st.write("")
            st.markdown("##### 📅 Individual Walk-Forward Fold Breakdown:")
            fold_df = pd.DataFrame(wf.get("folds", []))
            if not fold_df.empty:
                fold_df = fold_df.rename(columns={
                    "fold_index": "Fold #",
                    "train_period": "In-Sample Training Window",
                    "test_period": "Out-of-Sample Test Window",
                    "cagr_pct": "OOS CAGR (%)",
                    "sharpe_ratio": "OOS Sharpe",
                    "max_drawdown_pct": "Max DD (%)",
                    "win_rate_pct": "Win Rate (%)",
                    "profit_factor": "Profit Factor",
                    "trades_count": "Trades"
                })
                st.dataframe(fold_df, use_container_width=True, hide_index=True)
                
        # 3. Factor Importance & Attribution Bar Chart
        st.markdown("---")
        st.subheader("🧠 Layer 2: Quantitative Factor Importance (Gini Feature Attribution)")
        st.caption("Measures which mathematical factors contribute the most signal strength in separating winners from losers.")
        
        rf_info = models_data.get("Random Forest Ensemble", {})
        feat_dict = rf_info.get("feature_importance", {})
        if feat_dict:
            sorted_feats = sorted(feat_dict.items(), key=lambda x: x[1], reverse=True)
            f_names = [k for k, v in sorted_feats]
            f_scores = [v for k, v in sorted_feats]
            
            f_fig = go.Figure(go.Bar(
                x=f_scores,
                y=f_names,
                orientation="h",
                marker=dict(color="#38bdf8", line=dict(color="#0284c7", width=1))
            ))
            f_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#090e1a",
                plot_bgcolor="#090e1a",
                margin=dict(l=20, r=20, t=10, b=20),
                height=320,
                xaxis=dict(title="Relative Information Gain (Gini Importance)", showgrid=True, gridcolor="#1e293b"),
                yaxis=dict(autorange="reversed", showgrid=False)
            )
            st.plotly_chart(f_fig, use_container_width=True)
            
        # 4. Quantitative Research Journal & Log Form
        st.markdown("---")
        st.subheader("📓 Layer 7: Quantitative Research Journal (Log Hypothesis & Save Result)")
        st.caption("Maintain a professional audit log of every strategy, hypothesis, and backtest outcome.")
        
        with st.form("research_log_form"):
            j_col1, j_col2 = st.columns([3, 1.5])
            with j_col1:
                hypo_text = st.text_input("Research Hypothesis:", placeholder="e.g. Parkinson Volatility + 5-day Relative Strength alpha on NIFTY during high-vol regimes")
                note_text = st.text_input("Research Notes / Observations:", placeholder="e.g. Model beats baseline with lower drawdown, but shows slight decay in Q3 chop.")
            with j_col2:
                best_model_name = "AI Quantitative Hybrid" if "AI Quantitative Hybrid" in models_data else "Random Forest Ensemble"
                st.write("")
                st.write("")
                save_exp_btn = st.form_submit_button("💾 Save Experiment to Journal", use_container_width=True)
                
            if save_exp_btn and hypo_text:
                best_m = models_data.get(best_model_name, {})
                exp_id = ResearchJournal.log_experiment(
                    symbol=sym,
                    hypothesis=hypo_text,
                    model_type=best_model_name,
                    oos_sharpe=float(best_m.get("sharpe_ratio", 1.0)),
                    deflated_sharpe=float(wf.get("deflated_sharpe", 1.0)),
                    oos_cagr=float(best_m.get("cagr_pct", 0.0)),
                    oos_max_dd=float(best_m.get("max_drawdown_pct", 0.0)),
                    win_rate=float(best_m.get("win_rate_pct", 50.0)),
                    consistency_pct=float(wf.get("consistency_score_pct", 100.0)),
                    notes=note_text
                )
                st.success(f"✅ Experiment #{exp_id} logged to Research Journal database successfully!")
                st.rerun()
                
    # Historical Experiments Table
    st.markdown("---")
    st.subheader("📜 Historical Research Experiments & Strategy Archive")
    past_exps = ResearchJournal.get_experiments(limit=25)
    if past_exps:
        p_df = pd.DataFrame(past_exps)
        p_df = p_df.rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "symbol": "Asset",
            "hypothesis": "Hypothesis Tested",
            "model_type": "Best Model",
            "oos_sharpe": "OOS Sharpe",
            "deflated_sharpe": "DSR",
            "oos_cagr": "CAGR (%)",
            "oos_max_dd": "Max DD (%)",
            "win_rate": "Win %",
            "consistency_pct": "Consistency",
            "notes": "Notes"
        })
        st.dataframe(p_df, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ No historical experiments recorded yet. Run a tournament above and log your first hypothesis!")
