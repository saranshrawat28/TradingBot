"""
Paper Trading Accuracy Lab & Self-Diagnostic Benchmark Tab for Streamlit UI.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.paper_lab.paper_db import PaperDB
from src.paper_lab.lab_config import LabConfig
from src.paper_lab.daily_picker import DailyPicker
from src.paper_lab.chronological_evaluator import ChronologicalEvaluator
from src.paper_lab.report_generator import ReportGenerator
from src.data.data_fetcher import get_live_quote
from src.utils.helpers import get_ist_now, format_currency_inr

def render_paper_lab_tab(broker_instance=None):
    """Renders the Paper Trading Accuracy Lab dashboard."""
    now = get_ist_now()
    today_str = now.strftime("%Y-%m-%d")

    st.markdown("""
    <div style='display: flex; align-items: center; justify-content: space-between;'>
        <div>
            <h2 style='margin:0;'>🧪 Paper Trading Accuracy Lab & Self-Diagnostic Hub</h2>
            <div style='color: #94a3b8; font-size: 0.9rem; margin-top: 4px;'>
                Daily 5-Stock Quantitative Paper Engine • ₹1,00,000 Fixed Daily Capital • Chronological 1m Candle Replay • Automatic Failure Diagnostics
            </div>
        </div>
        <div>
            <span style='background: #1e293b; color: #38bdf8; border: 1px solid #0284c7; padding: 6px 14px; border-radius: 8px; font-weight: 600; font-size: 0.85rem;'>
                ACTIVE ENGINE: """ + LabConfig.CONFIG_VERSION + """
            </span>
        </div>
    </div>
    <hr style='margin: 16px 0; border-color: #334155;'/>
    """, unsafe_allow_html=True)

    # 1. Action Buttons Control Bar
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.5, 1.8, 1.8, 2])
    with btn_col1:
        if st.button("⚡ Run Scan & Fills Now", use_container_width=True):
            with st.spinner("Scanning universe & generating Top 5 picks..."):
                DailyPicker.run_daily_picker_catchup()
            st.success("Today's picks generated & confirmed!")
            st.rerun()

    with btn_col2:
        if st.button("🔍 Run Outcome Replay", use_container_width=True):
            with st.spinner("Replaying 1m/5m candles chronologically..."):
                ChronologicalEvaluator.evaluate_all_picks_for_date()
            st.success("Chronological outcome evaluation complete!")
            st.rerun()

    with btn_col3:
        if st.button("📝 Generate 7-Day Report", use_container_width=True):
            with st.spinner("Compiling accuracy & diagnostic audit..."):
                ReportGenerator.generate_report(days_lookback=7)
            st.success("7-Day report generated!")
            st.rerun()

    with btn_col4:
        if st.button("📊 Generate 28-Day Benchmark", use_container_width=True):
            with st.spinner("Compiling 4-week rolling audit..."):
                ReportGenerator.generate_report(days_lookback=28)
            st.success("28-Day Benchmark report generated!")
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    # 2. Cumulative KPI Cards (Past 7 Days)
    recent_report = ReportGenerator.generate_report(days_lookback=7)
    fin = recent_report.get("financial_summary", {})
    acc = recent_report.get("prediction_accuracy", {})

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        notional = fin.get("total_notional_deployed_rs", 0.0)
        st.metric("Notional Deployed (7d)", f"₹{notional:,.0f}", f"{acc.get('total_picks', 0)} Picks")
    with kpi2:
        pnl = fin.get("net_realized_pnl_rs", 0.0)
        ret = fin.get("net_return_pct", 0.0)
        pnl_color = "normal" if pnl >= 0 else "inverse"
        st.metric("Net Realized P&L (7d)", f"{'+' if pnl>=0 else ''}₹{pnl:,.2f}", f"{ret:+.2f}% on Capital")
    with kpi3:
        wr = acc.get("win_rate_pct", 0.0)
        st.metric("Prediction Win Rate", f"{wr:.1f}%", f"{acc.get('winning_picks', 0)} Wins / {acc.get('losing_picks', 0)} Losses")
    with kpi4:
        pf = fin.get("profit_factor", 1.0)
        st.metric("Profit Factor", f"{pf:.2f}", f"Avg Win: ₹{fin.get('avg_winner_rs', 0):,.0f}")
    with kpi5:
        t1_c = acc.get("t1_hit_count", 0)
        t2_c = acc.get("t2_hit_count", 0)
        sl_c = acc.get("sl_hit_count", 0)
        st.metric("Target vs Stop Hits", f"{t1_c + t2_c} 🎯 vs {sl_c} 🛑", f"T1: {t1_c} | T2: {t2_c}")

    st.markdown("---")

    # 3. Today's Active Recommendations Table
    st.markdown(f"### 📋 Today's Recommendations & Live Execution Board (`{today_str}`)")

    today_picks = PaperDB.get_picks_by_date(today_str)
    if not today_picks:
        st.markdown(f"""
        <div style='background: #0f172a; border: 1.5px solid #1e293b; border-left: 4px solid #38bdf8; border-radius: 10px; padding: 18px 22px; margin: 10px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <div style='font-size: 1.05rem; font-weight: 800; color: #f8fafc;'>🚀 Armed & Ready for Monday Market Session (08:50 AM IST)</div>
                    <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 4px;'>
                        The 24/7 background scheduler is active. It will automatically scan 200+ Indian stocks at 08:50 AM, fill 5 paper trades at 09:15 AM with ₹1,00,000 dummy capital, and evaluate results at 03:35 PM.
                    </div>
                </div>
                <span style='background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 6px 12px; border-radius: 8px; font-weight: 700; font-size: 0.80rem;'>🟢 SCHEDULER ACTIVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        table_rows = []
        for p in today_picks:
            sym = p["symbol"]
            ep = float(p.get("entry_price") or p.get("signal_price", 100.0))
            qty = int(p.get("quantity", 1))

            # Fetch live quote for real-time unrealized P&L
            quote = get_live_quote(sym)
            curr_p = float(quote.get("price", ep)) if quote.get("price") else ep
            unrealized_rs = (curr_p - ep) * qty
            unrealized_pct = ((curr_p - ep) / ep) * 100.0 if ep > 0 else 0.0

            status = p.get("status", "ACTIVE")
            status_badge = "🟢 ACTIVE" if status == "ACTIVE" else ("⏳ PENDING OPEN" if status == "PENDING_OPEN" else "🔒 CLOSED")

            table_rows.append({
                "Symbol": f"**{sym}**",
                "Company": p.get("display_name", sym),
                "Score": f"⭐ {p['advisor_score']:.1f}/10",
                "Entry Fill (₹)": f"₹{ep:,.2f}",
                "Live LTP (₹)": f"₹{curr_p:,.2f}",
                "Target 1 (₹)": f"₹{p['target_1']:,.2f}",
                "Target 2 (₹)": f"₹{p['target_2']:,.2f}",
                "Stop-Loss (₹)": f"₹{p['stop_loss']:,.2f}",
                "Qty / Cap": f"{qty} shares (₹{p['allocated_capital']:,.0f})",
                "Live P&L (₹)": f"{'+' if unrealized_rs >= 0 else ''}₹{unrealized_rs:,.2f} ({unrealized_pct:+.2f}%)",
                "Status": status_badge
            })

        df_picks = pd.DataFrame(table_rows)
        st.dataframe(df_picks, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 4. Signal Diagnostic Failure Breakdown
    st.markdown("### 🔬 Self-Diagnostic Failure Radar (Where Recommender Lagged)")

    diag = recent_report.get("signal_diagnostics", {})
    losers_cnt = diag.get("total_failures_analyzed", 0)

    if losers_cnt == 0:
        st.success("🌟 Zero losing recommendations recorded in the last 7 days! No failure diagnostics to display.")
    else:
        d_col1, d_col2 = st.columns([1.2, 1.8])
        with d_col1:
            st.markdown(f"**Failure Analysis on `{losers_cnt}` Losing Setups:**")
            st.markdown(f"• **RSI Overbought (>65)**: `{diag.get('rsi_fail_count', 0)}/{losers_cnt}` ({diag.get('rsi_fail_pct', 0)}%)")
            st.markdown(f"• **Weak Volume (<1.0 RVol)**: `{diag.get('rvol_fail_count', 0)}/{losers_cnt}` ({diag.get('rvol_fail_pct', 0)}%)")
            st.markdown(f"• **Late VWAP Chasing (>0.4σ)**: `{diag.get('vwap_fail_count', 0)}/{losers_cnt}` ({diag.get('vwap_fail_pct', 0)}%)")
            st.markdown(f"• **Low ADX Chop (<20)**: `{diag.get('adx_fail_count', 0)}/{losers_cnt}` ({diag.get('adx_fail_pct', 0)}%)")

            if recent_report.get("sample_warning"):
                st.caption(recent_report["sample_warning"])

        with d_col2:
            fig = go.Figure(data=[
                go.Bar(
                    x=["RSI Overbought", "Weak Volume", "Late VWAP Entry", "Low ADX Chop"],
                    y=[
                        diag.get('rsi_fail_pct', 0),
                        diag.get('rvol_fail_pct', 0),
                        diag.get('vwap_fail_pct', 0),
                        diag.get('adx_fail_pct', 0)
                    ],
                    marker_color=["#ef4444", "#f97316", "#eab308", "#8b5cf6"],
                    text=[f"{v:.1f}%" for v in [
                        diag.get('rsi_fail_pct', 0),
                        diag.get('rvol_fail_pct', 0),
                        diag.get('vwap_fail_pct', 0),
                        diag.get('adx_fail_pct', 0)
                    ]],
                    textposition="auto"
                )
            ])
            fig.update_layout(
                title="Losing Trade Failure Mode Frequency (%)",
                template="plotly_dark",
                height=260,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(title="% of Losers", range=[0, 100])
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 💡 Engine Optimization Recommendations:")
        for rec in recent_report.get("recommendations", []):
            st.markdown(f"• {rec}")

    st.markdown("---")

    # 5. Full Report History & Markdown Viewer
    st.markdown("### 📜 Weekly Diagnostic Reports Archive")
    saved_reports = ReportGenerator.list_saved_reports()

    if not saved_reports:
        st.info("No saved reports in storage yet. Click **'Generate 7-Day Report'** to create your first report.")
    else:
        selected_rep = st.selectbox(
            "Select Historical Report to View:",
            [f"{r['file_name']} — {r['title']} (Win Rate: {r['win_rate']}%)" for r in saved_reports]
        )

        chosen_idx = [f"{r['file_name']} — {r['title']} (Win Rate: {r['win_rate']}%)" for r in saved_reports].index(selected_rep)
        chosen_path = saved_reports[chosen_idx]["file_path"]

        try:
            import json
            with open(chosen_path, "r", encoding="utf-8") as f:
                rep_json = json.load(f)
            st.markdown(rep_json.get("markdown_text", ""))
        except Exception as e:
            st.error(f"Could not load report: {e}")
