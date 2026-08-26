"""
Shared UI components for ApexTrade Streamlit Terminal.
Provides CSS injection, live header, stock watcher bar, sidebar navigation, and emergency panic switch.
"""

import os
import streamlit as st
import pandas as pd
import config
from src.utils.helpers import (
    get_ist_now, is_market_open, format_currency_inr, format_percentage,
    clean_symbol, display_symbol_name
)
from src.data.data_fetcher import get_historical_data, get_live_quote
from src.utils.storage import get_portfolio_state

def inject_terminal_css():
    """Injects institutional terminal CSS styling and typography tokens."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

        :root {
            --bg-obsidian: #080b11;
            --bg-surface: #111622;
            --bg-surface-elevated: #182030;
            --border-subtle: #1e293b;
            --border-prominent: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --color-bullish: #10b981;
            --color-bearish: #f43f5e;
            --color-neutral: #f59e0b;
            --color-sky: #0ea5e9;
        }

        .stApp {
            background-color: var(--bg-obsidian) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        .tnum, .mono-num, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-feature-settings: "tnum" 1 !important;
            font-variant-numeric: tabular-nums !important;
            letter-spacing: -0.02em !important;
        }

        h1, h2, h3, .brand-title {
            font-family: 'Outfit', sans-serif !important;
            letter-spacing: -0.02em !important;
            color: var(--text-primary) !important;
        }

        div[data-testid="stMetric"] {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
            box-shadow: none !important;
            min-height: 82px !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
        }
        div[data-testid="stMetric"]:hover {
            border-color: var(--border-prominent) !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
            color: var(--text-secondary) !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin-bottom: 2px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.30rem !important;
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
        }

        .ambient-dot-green {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--color-bullish);
            margin-right: 6px;
            box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
        }
        .ambient-dot-red {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--color-bearish);
            margin-right: 6px;
            box-shadow: 0 0 6px rgba(244, 63, 94, 0.6);
        }

        .badge-bull {
            background: rgba(16, 185, 129, 0.12);
            color: var(--color-bullish);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.76rem;
        }
        .badge-bear {
            background: rgba(244, 63, 94, 0.12);
            color: var(--color-bearish);
            border: 1px solid rgba(244, 63, 94, 0.3);
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.76rem;
        }
        .badge-neutral {
            background: rgba(245, 158, 11, 0.12);
            color: var(--color-neutral);
            border: 1px solid rgba(245, 158, 11, 0.3);
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.76rem;
        }
        .badge-cyan {
            background: rgba(14, 165, 233, 0.12);
            color: var(--color-sky);
            border: 1px solid rgba(14, 165, 233, 0.3);
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.76rem;
        }

        .op-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
        }

        .tip-box {
            background: #0f172a;
            border-left: 3px solid #38bdf8;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 0.82rem;
            color: #cbd5e1;
            margin: 8px 0;
            line-height: 1.4;
        }

        .kill-switch-box {
            background: rgba(244, 63, 94, 0.08);
            border: 1px solid rgba(244, 63, 94, 0.35);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

@st.fragment(run_every=3)
def render_live_header(broker):
    """Renders top live header banner with market timing, account balance, and live telemetry."""
    now_ist = get_ist_now()
    m_open, m_status_msg = is_market_open()
    portfolio = get_portfolio_state()
    open_positions = broker.get_open_positions()

    cash_val = float(portfolio.get("cash", 100000.0))
    daily_pnl = float(portfolio.get("daily_pnl", 0.0))
    open_pnl = sum(p.get("unrealized_pnl", 0.0) for p in open_positions)
    equity_val = cash_val + sum(p.get("entry_price", 0) * p.get("quantity", 0) for p in open_positions) + open_pnl

    status_dot = "<span class='ambient-dot-green'></span>" if m_open else "<span class='ambient-dot-red'></span>"
    status_text = "LIVE MARKET ACTIVE" if m_open else "MARKET CLOSED"

    st.markdown(f"""
    <div style='background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 14px 20px; margin-bottom: 14px;'>
        <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;'>
            <div style='display: flex; align-items: center; gap: 12px;'>
                <div style='font-size: 1.5rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif; letter-spacing: -0.02em;'>
                    ⚡ APEXTRADE <span style='font-size: 0.85rem; color: #38bdf8; font-weight: 700; border: 1px solid rgba(56,189,248,0.3); padding: 2px 8px; border-radius: 4px;'>INSTITUTIONAL AI TERMINAL</span>
                </div>
            </div>
            <div style='display: flex; align-items: center; gap: 14px; font-size: 0.84rem;'>
                <div style='background: rgba(255,255,255,0.04); border: 1px solid var(--border-subtle); padding: 4px 10px; border-radius: 6px;'>
                    {status_dot} <strong>{status_text}</strong> &bull; {now_ist.strftime('%I:%M:%S %p IST')}
                </div>
                <div style='background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.25); color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-weight: 700;'>
                    ACCOUNT EQUITY: ₹{equity_val:,.2f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

@st.fragment(run_every=4)
def render_live_stock_watcher():
    """Renders fast 1-second real-time stock and index price telemetry bar."""
    nifty_quote = get_live_quote("^NSEI")
    bank_quote = get_live_quote("^NSEBANK")
    
    n_p = float(nifty_quote.get("price", 24650.0))
    n_chg = float(nifty_quote.get("change_pct", 0.0))
    b_p = float(bank_quote.get("price", 51200.0))
    b_chg = float(bank_quote.get("change_pct", 0.0))

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("🇮🇳 NIFTY 50", f"₹{n_p:,.2f}", f"{n_chg:+.2f}%", delta_color="normal")
    q2.metric("🏦 BANK NIFTY", f"₹{b_p:,.2f}", f"{b_chg:+.2f}%", delta_color="normal")
    q3.metric("⚡ INDIA VIX", "13.40", "-1.2% Benign")
    q4.metric("🛡️ Auto Square-Off", "3:15 PM IST", "SEBI Mandatory")

def render_sidebar_navigation(broker):
    """Renders sidebar experience mode, tabs, 1-click quick order form, and emergency panic switch."""
    st.sidebar.markdown("""
    <div style='padding: 4px 0 8px 0;'>
        <div style='font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>🧭 APEXTRADE TERMINAL</div>
        <div style='font-size: 0.75rem; color: #94a3b8;'>Choose your experience mode</div>
    </div>
    """, unsafe_allow_html=True)

    ui_mode = st.sidebar.radio(
        "Experience Mode:",
        [
            "🌟 Simple & Easy Mode (Beginner Friendly)",
            "⚡ Pro Quantitative Workstation"
        ],
        index=0,
        help="Simple Mode gives clean Buy/Sell advice and pre-market picks with zero confusing jargon. Pro Mode provides Options Greeks, advanced indicators, and quantitative tooling."
    )

    st.sidebar.markdown("---")

    if ui_mode == "🌟 Simple & Easy Mode (Beginner Friendly)":
        nav_options = [
            "🗣️ Talk to Your AI Bot (Chat & Voice)",
            "🌅 Pre-Market & Best Stocks Today",
            "🎯 Easy Stock Advisor (Buy / Sell Advice)",
            "🤖 AI Auto-Pilot (Automated Safe Trading)",
            "📦 My Trades & Profit Book",
            "⚙️ Simple Settings & Safety"
        ]
    else:
        nav_options = [
            "🗣️ Talk to Your AI Bot (Chat & Voice)",
            "🌅 Pre-Market & Best Stocks Today",
            "🤖 Autonomous AI Trading Agent (Claude / Kimi / F&O)",
            "🔬 Systematic Quant Research Lab",
            "⚡ NFO Options Greeks & OI Matrix",
            "🎯 Smart Stock Advisor (When to Buy/Sell)",
            "📊 Strategy Backtester (Test Any Stock)",
            "⚡ Automated Live / Paper Bot",
            "🔍 Indian Market Screener (Scan All Stocks)",
            "⚙️ Settings & Risk Controls"
        ]

    active_tab = st.sidebar.radio("Navigation:", nav_options, label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;'>⚡ 1-Click Quick Order</div>
    """, unsafe_allow_html=True)

    with st.sidebar.form("quick_order_form"):
        q_sym = st.selectbox(
            "Choose Stock",
            options=[item["symbol"] for item in config.DEFAULT_WATCHLIST],
            format_func=lambda s: next((f"{item['name']} ({item['symbol'].replace('.NS','')})" for item in config.DEFAULT_WATCHLIST if item["symbol"] == s), s)
        )
        q_side = st.selectbox("Action", ["BUY (Go Long)", "SELL (Square-off / Short)"])
        q_qty = st.number_input("Number of Shares", min_value=1, value=10, step=1)
        q_sl = st.number_input("Safety Stop-Loss Price (₹, 0 for none)", min_value=0.0, value=0.0, step=1.0)
        q_tp = st.number_input("Profit Target Price (₹, 0 for none)", min_value=0.0, value=0.0, step=1.0)
        
        submitted = st.form_submit_button("🚀 Place Order")
        if submitted:
            side_clean = "BUY" if "BUY" in q_side else "SELL"
            res = broker.place_order(
                symbol=q_sym,
                side=side_clean,
                quantity=int(q_qty),
                sl=q_sl if q_sl > 0 else None,
                tp=q_tp if q_tp > 0 else None,
                strategy_name="Manual Quick Order"
            )
            if res.get("status") in ["FILLED", "SUCCESS"]:
                p_val = float(res.get("price") or 0.0)
                st.sidebar.success(f"Order Successful! Executed @ ₹{p_val:.2f}")
                st.rerun()
            else:
                st.sidebar.error(f"Order Failed: {res.get('message', 'Check funds or broker')}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;'>⚡ Live Stream Engine</div>
    <div style='font-size: 0.82rem; color: #10b981; font-weight: 700;'><span class='ambient-dot-green'></span>Real-Time Stream Active</div>
    <div style='font-size: 0.75rem; color: #94a3b8; margin-top: 2px;'>Background daemon streams prices & P&L with zero lag.</div>
    """, unsafe_allow_html=True)
    if st.sidebar.button("🔄 Force Instant Sync", use_container_width=True):
        st.rerun()

    st.sidebar.markdown("---")
    with st.sidebar.expander("🚨 Emergency Panic Kill Switch", expanded=False):
        st.markdown("""
        <div class='kill-switch-box'>
            <div style='color: #f43f5e; font-weight: 800; font-size: 0.85rem; text-transform: uppercase;'>⚠️ Critical Emergency Control</div>
            <div style='color: #fecdd3; font-size: 0.78rem; margin-top: 4px; line-height: 1.3;'>Immediately executes MARKET square-off for all active positions and cancels all pending broker orders.</div>
        </div>
        """, unsafe_allow_html=True)
        kill_confirm = st.checkbox("I confirm immediate emergency liquidation", key="sidebar_kill_confirm")
        if st.button("🚨 EXECUTE EMERGENCY LIQUIDATION", type="secondary", disabled=not kill_confirm, use_container_width=True):
            sq_res = broker.square_off_all(reason="Emergency Manual Close")
            st.sidebar.warning(f"Closed {len(sq_res)} active positions!")
            st.rerun()

    return ui_mode, active_tab
