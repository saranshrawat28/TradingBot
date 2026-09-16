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

    # 0. Virtual Paper Trading Funds & Account Wallet State
    if "paper_funds_total" not in st.session_state:
        st.session_state["paper_funds_total"] = LabConfig.TOTAL_DAILY_CAPITAL
    if "paper_funds_per_pick" not in st.session_state:
        st.session_state["paper_funds_per_pick"] = LabConfig.DAILY_CAPITAL_PER_PICK

    total_paper_cap = float(st.session_state["paper_funds_total"])
    per_pick_cap = float(st.session_state["paper_funds_per_pick"])

    # Calculate portfolio funds and active MTM
    portfolio_summary = PaperDB.get_portfolio_summary()
    all_time_realized_pnl = float(portfolio_summary.get("total_realized_pnl", 0.0))
    all_time_trades = portfolio_summary.get("total_trades", 0)
    all_time_wr = portfolio_summary.get("win_rate_pct", 0.0)

    today_picks = PaperDB.get_picks_by_date(today_str)
    active_picks = [p for p in today_picks if p.get("status") == "ACTIVE"]
    deployed_margin = sum(float(p.get("allocated_capital", per_pick_cap)) for p in active_picks)
    available_free_cash = max(0.0, total_paper_cap - deployed_margin)

    # Calculate live intraday unrealized P&L
    live_unrealized_pnl = 0.0
    for p in active_picks:
        sym = p["symbol"]
        ep = float(p.get("entry_price") or p.get("signal_price", 100.0))
        qty = int(p.get("quantity", 1))
        q = get_live_quote(sym)
        curr_p = float(q.get("price", ep)) if q.get("price") else ep
        live_unrealized_pnl += (curr_p - ep) * qty

    total_equity_nav = total_paper_cap + all_time_realized_pnl + live_unrealized_pnl
    total_roi_pct = ((total_equity_nav - total_paper_cap) / total_paper_cap * 100.0) if total_paper_cap > 0 else 0.0

    # 💼 Paper Trading Funds & Margin Wallet Card
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #0b132b 0%, #1c2541 50%, #1e1b4b 100%); border: 1.5px solid #3b82f6; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;'>
        <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 12px;'>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <span style='font-size: 1.25rem;'>💼</span>
                <div>
                    <span style='font-size: 1.05rem; font-weight: 800; color: #f8fafc;'>ApexTrade Virtual Paper Account & Capital Wallet</span>
                    <span style='margin-left: 10px; font-size: 0.75rem; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 2px 8px; border-radius: 6px; font-weight: 700;'>ZERO REAL RISK</span>
                </div>
            </div>
            <div style='color: #94a3b8; font-size: 0.80rem;'>
                Allocated Capital: <b>₹{per_pick_cap:,.0f} / Trade</b> (Max 5 Daily Picks)
            </div>
        </div>
        <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;'>
            <div style='background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);'>
                <div style='color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;'>Virtual Equity (NAV)</div>
                <div style='color: #38bdf8; font-size: 1.25rem; font-weight: 800; margin-top: 2px;'>₹{total_equity_nav:,.2f}</div>
                <div style='color: {'#10b981' if total_roi_pct >= 0 else '#ef4444'}; font-size: 0.75rem; font-weight: 600;'>{total_roi_pct:+.2f}% All-Time ROI</div>
            </div>
            <div style='background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);'>
                <div style='color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;'>Available Free Cash</div>
                <div style='color: #f8fafc; font-size: 1.25rem; font-weight: 800; margin-top: 2px;'>₹{available_free_cash:,.2f}</div>
                <div style='color: #94a3b8; font-size: 0.75rem;'>Ready for Orders</div>
            </div>
            <div style='background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);'>
                <div style='color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;'>Active Deployed Margin</div>
                <div style='color: #fbbf24; font-size: 1.25rem; font-weight: 800; margin-top: 2px;'>₹{deployed_margin:,.2f}</div>
                <div style='color: #94a3b8; font-size: 0.75rem;'>{len(active_picks)} Active Positions</div>
            </div>
            <div style='background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);'>
                <div style='color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;'>Realized P&L</div>
                <div style='color: {'#10b981' if all_time_realized_pnl >= 0 else '#ef4444'}; font-size: 1.25rem; font-weight: 800; margin-top: 2px;'>{'+' if all_time_realized_pnl > 0 else ''}₹{all_time_realized_pnl:,.2f}</div>
                <div style='color: #94a3b8; font-size: 0.75rem;'>{all_time_trades} Closed ({all_time_wr:.1f}% WR)</div>
            </div>
            <div style='background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);'>
                <div style='color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;'>Live Intraday P&L (MTM)</div>
                <div style='color: {'#10b981' if live_unrealized_pnl >= 0 else '#ef4444'}; font-size: 1.25rem; font-weight: 800; margin-top: 2px;'>{'+' if live_unrealized_pnl >= 0 else ''}₹{live_unrealized_pnl:,.2f}</div>
                <div style='color: #38bdf8; font-size: 0.75rem;'>⚡ Real-time Ticks</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Wallet Funds Customization Expander
    with st.expander("⚙️ Manage Virtual Paper Funds & Capital Settings"):
        f_col1, f_col2, f_col3 = st.columns([2, 2, 1.5])
        with f_col1:
            new_total = st.selectbox(
                "Starting Virtual Paper Balance:",
                [50000.0, 100000.0, 200000.0, 500000.0, 1000000.0],
                index=[50000.0, 100000.0, 200000.0, 500000.0, 1000000.0].index(total_paper_cap) if total_paper_cap in [50000.0, 100000.0, 200000.0, 500000.0, 1000000.0] else 1,
                format_func=lambda x: f"₹{x:,.0f}"
            )
            if new_total != total_paper_cap:
                st.session_state["paper_funds_total"] = new_total
                st.rerun()

        with f_col2:
            new_per_pick = st.selectbox(
                "Capital Allocation Per Stock Pick:",
                [10000.0, 20000.0, 50000.0, 100000.0],
                index=[10000.0, 20000.0, 50000.0, 100000.0].index(per_pick_cap) if per_pick_cap in [10000.0, 20000.0, 50000.0, 100000.0] else 1,
                format_func=lambda x: f"₹{x:,.0f} / stock"
            )
            if new_per_pick != per_pick_cap:
                st.session_state["paper_funds_per_pick"] = new_per_pick
                st.rerun()

        with f_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset Paper Balance", use_container_width=True):
                st.session_state["paper_funds_total"] = 100000.0
                st.session_state["paper_funds_per_pick"] = 20000.0
                st.success("Paper funds reset to ₹1,00,000 baseline!")
                st.rerun()

    # 1. Action Buttons Control Bar
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1.4, 1.6, 1.6, 1.6, 1.8])
    with btn_col1:
        if st.button("⚡ Run Daily Fills", use_container_width=True):
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
        if st.button("📅 Generate Daily Report", use_container_width=True):
            with st.spinner("Compiling Today's Daily Performance Report..."):
                ReportGenerator.generate_report(days_lookback=1)
            st.success("Today's Daily Report generated!")
            st.rerun()

    with btn_col4:
        if st.button("📝 Generate 7-Day Report", use_container_width=True):
            with st.spinner("Compiling accuracy & diagnostic audit..."):
                ReportGenerator.generate_report(days_lookback=7)
            st.success("7-Day report generated!")
            st.rerun()

    with btn_col5:
        if st.button("📊 Generate 28-Day Benchmark", use_container_width=True):
            with st.spinner("Compiling 4-week rolling audit..."):
                ReportGenerator.generate_report(days_lookback=28)
            st.success("28-Day Benchmark report generated!")
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    # 2. Performance Scope Selector & Excel/CSV Export Buttons
    sel_col1, sel_col2, sel_col3 = st.columns([2.5, 1.2, 1.0])
    with sel_col1:
        scope = st.radio("📈 Select Report & Metrics Scope:", ["📅 Today's Daily Returns (1-Day)", "📊 Rolling 7-Day Performance", "🏛️ Rolling 28-Day Benchmark"], horizontal=True)
    
    days_lb = 1 if "1-Day" in scope else (28 if "28-Day" in scope else 7)
    scope_tag = "Today" if days_lb == 1 else f"{days_lb}d"
    recent_report = ReportGenerator.generate_report(days_lookback=days_lb)

    # Generate in-memory Excel and CSV bytes for download
    try:
        excel_bytes = ReportGenerator.export_report_to_excel_bytes(recent_report)
    except Exception:
        excel_bytes = b""
    try:
        csv_str = ReportGenerator.export_report_to_csv_string(recent_report)
    except Exception:
        csv_str = ""

    with sel_col2:
        if excel_bytes:
            st.download_button(
                label=f"📥 Download Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"ApexTrade_Report_{today_str}_{scope_tag}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with sel_col3:
        if csv_str:
            st.download_button(
                label=f"📥 CSV (.csv)",
                data=csv_str,
                file_name=f"ApexTrade_Report_{today_str}_{scope_tag}.csv",
                mime="text/csv",
                use_container_width=True
            )

    fin = recent_report.get("financial_summary", {})
    acc = recent_report.get("prediction_accuracy", {})

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        notional = fin.get("total_notional_deployed_rs", 0.0)
        st.metric(f"Notional Deployed ({scope_tag})", f"₹{notional:,.0f}", f"{acc.get('total_picks', 0)} Picks")
    with kpi2:
        pnl = fin.get("net_realized_pnl_rs", 0.0)
        ret = fin.get("net_return_pct", 0.0)
        pnl_color = "normal" if pnl >= 0 else "inverse"
        st.metric(f"Net Realized P&L ({scope_tag})", f"{'+' if pnl>0 else ''}₹{pnl:,.2f}", f"{ret:+.2f}% on Capital")
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
    day_name = now.strftime("%A")
    if not today_picks:
        is_live_hour = (now.hour == 9 and now.minute >= 15) or (10 <= now.hour < 15) or (now.hour == 15 and now.minute <= 30)
        session_text = f"Live {day_name} Market Session" if is_live_hour else f"Upcoming {day_name} Market Session"
        st.markdown(f"""
        <div style='background: #0f172a; border: 1.5px solid #1e293b; border-left: 4px solid #38bdf8; border-radius: 10px; padding: 18px 22px; margin: 10px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <div style='font-size: 1.05rem; font-weight: 800; color: #f8fafc;'>🚀 Armed & Ready for {session_text} (08:50 AM IST)</div>
                    <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 4px;'>
                        The 24/7 background scheduler is active. It scans 200+ Indian stocks at 08:50 AM, fills 5 paper trades at 09:15 AM with ₹{total_paper_cap:,.0f} virtual capital (₹{per_pick_cap:,.0f} / trade), and evaluates results at 03:35 PM. Click <b>⚡ Run Daily Fills</b> to generate on demand.
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
            from pathlib import Path
            with open(chosen_path, "r", encoding="utf-8") as f:
                rep_json = json.load(f)

            dl_c1, dl_c2 = st.columns([1, 1])
            try:
                hist_xlsx = ReportGenerator.export_report_to_excel_bytes(rep_json)
                with dl_c1:
                    st.download_button(
                        label=f"📥 Download Report in Excel (.xlsx)",
                        data=hist_xlsx,
                        file_name=f"{Path(chosen_path).stem}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            except Exception:
                pass

            try:
                hist_csv = ReportGenerator.export_report_to_csv_string(rep_json)
                with dl_c2:
                    st.download_button(
                        label=f"📥 Download Report in CSV (.csv)",
                        data=hist_csv,
                        file_name=f"{Path(chosen_path).stem}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception:
                pass

            st.markdown(rep_json.get("markdown_text", ""))
        except Exception as e:
            st.error(f"Could not load report: {e}")
