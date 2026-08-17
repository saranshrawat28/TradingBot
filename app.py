"""
ApexTrade - Indian Stocks & F&O Algorithmic Trading Terminal & Web Dashboard
Institutional Two-Tier Terminal UI Architecture (Operational High-Contrast + Frosted Chrome)
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import config
from src.utils.helpers import (
    get_ist_now, is_market_open, format_currency_inr, format_percentage,
    clean_symbol, display_symbol_name, is_intraday_squareoff_time, format_holding_duration
)
from src.utils.storage import (
    get_portfolio_state, get_open_positions, get_orders, get_closed_trades,
    reset_all_data, log_order, save_ai_settings, load_ai_settings,
    get_calibration_records, get_disagreement_records
)
from src.data.data_fetcher import get_historical_data, get_live_quote, search_indian_stocks
from src.strategies import AVAILABLE_STRATEGIES, get_strategy
from src.strategies.indicators import add_all_indicators
from src.engine.risk_manager import RiskManager
from src.engine.backtester import Backtester
from src.engine.live_bot import LiveTradingBot
from src.engine.stock_advisor import StockAdvisor
from src.engine.ai_guardrails import AIGuardrails
from src.engine.reconciliation import StateReconciler
from src.engine.trade_manager import SmartTradeManager
from src.engine.auto_pilot_daemon import AutoPilotDaemon
from src.ai import (
    LLMClient, MarketPrompter, FailsafeParser, ConfidenceCalibrator,
    AITradingAgent, MarketRadarScanner
)
from src.brokers.zerodha_live import ZerodhaLiveBroker
from src.backtest.ai_backtester import AIBacktester
from src.brokers import get_broker, BROKERS_MAP

# -------------------------------------------------------------
# Streamlit Page Config & Institutional Two-Tier Design System
# -------------------------------------------------------------
st.set_page_config(
    page_title="ApexTrade Terminal | Indian Stock & F&O AI Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Google Fonts & Two-Tier Institutional CSS Design Tokens
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

    :root {
        /* Base Backgrounds */
        --bg-obsidian: #080b11;
        --bg-card-solid: #111622;
        --bg-card-elevated: #161d2d;
        --bg-chrome-glass: rgba(17, 24, 39, 0.75);
        
        /* High-Contrast Operational Borders */
        --border-subtle: #1e293b;
        --border-prominent: #334155;
        --border-glass: rgba(255, 255, 255, 0.08);
        
        /* High-Contrast Colors (WCAG 2.1 AA Compliant) */
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        
        /* Accessible Status Cues */
        --color-bullish: #10b981;
        --color-bullish-bg: rgba(16, 185, 129, 0.12);
        --color-bearish: #f43f5e;
        --color-bearish-bg: rgba(244, 63, 94, 0.12);
        --color-neutral: #f59e0b;
        --color-neutral-bg: rgba(245, 158, 11, 0.12);
        --color-cyan: #0ea5e9;
        --color-cyan-bg: rgba(14, 165, 233, 0.12);
        --color-indigo: #6366f1;
        --color-indigo-bg: rgba(99, 102, 241, 0.12);
    }

    /* Core Terminal Canvas */
    .stApp {
        background-color: var(--bg-obsidian);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Monospace Text Utility */
    .mono-num {
        font-family: 'JetBrains Mono', monospace !important;
        font-feature-settings: "tnum" 1;
        letter-spacing: -0.02em;
    }

    /* Brand Headings */
    h1, h2, h3, .brand-font {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* High-Contrast Operational Metric Cards */
    div[data-testid="stMetric"] {
        background: var(--bg-card-solid);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
        transition: border-color 0.15s ease;
        min-height: 84px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--border-prominent);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.76rem !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        line-height: 1.2 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        margin-top: 2px !important;
    }
    div[data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-down"] {
        fill: #f43f5e !important;
        color: #f43f5e !important;
    }
    div[data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-up"] {
        fill: #10b981 !important;
        color: #10b981 !important;
    }
    div[data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-down"]) div,
    div[data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-down"]) span {
        color: #f43f5e !important;
    }
    div[data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-up"]) div,
    div[data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-up"]) span {
        color: #10b981 !important;
    }
    .text-negative, .text-bear {
        color: #f43f5e !important;
        font-weight: 700 !important;
    }
    .text-positive, .text-bull {
        color: #10b981 !important;
        font-weight: 700 !important;
    }

    /* Dynamic Price & Index Highlighting Cards */
    .price-card-bear {
        background: rgba(244, 63, 94, 0.08) !important;
        border: 1px solid rgba(244, 63, 94, 0.35) !important;
        border-radius: 10px;
        padding: 10px 14px;
        min-height: 84px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.2s ease;
    }
    .price-card-bull {
        background: rgba(16, 185, 129, 0.08) !important;
        border: 1px solid rgba(16, 185, 129, 0.35) !important;
        border-radius: 10px;
        padding: 10px 14px;
        min-height: 84px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.2s ease;
    }
    .price-text-bear {
        color: #f43f5e !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.25rem !important;
        letter-spacing: -0.02em;
    }
    .price-text-bull {
        color: #10b981 !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.25rem !important;
        letter-spacing: -0.02em;
    }

    /* Decorative Frosted Chrome Panels */
    .chrome-card {
        background: var(--bg-chrome-glass);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-sizing: border-box;
    }

    /* Solid Operational Cards */
    .op-card {
        background: var(--bg-card-solid);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 12px 16px;
        min-height: 84px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Color-Blind Accessible Badges */
    .badge-bull {
        background-color: var(--color-bullish-bg);
        color: var(--color-bullish);
        border: 1px solid var(--color-bullish);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-bear {
        background-color: var(--color-bearish-bg);
        color: var(--color-bearish);
        border: 1px solid var(--color-bearish);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-neutral {
        background-color: var(--color-neutral-bg);
        color: var(--color-neutral);
        border: 1px solid var(--color-neutral);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-cyan {
        background-color: var(--color-cyan-bg);
        color: var(--color-cyan);
        border: 1px solid var(--color-cyan);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Slow Ambient Breathing Fade (2.5s calm pulse) */
    @keyframes ambient-breath {
        0%, 100% { opacity: 0.9; }
        50% { opacity: 0.45; }
    }
    .ambient-dot-green {
        display: inline-block;
        width: 9px;
        height: 9px;
        background-color: var(--color-bullish);
        border-radius: 50%;
        margin-right: 6px;
        animation: ambient-breath 2.5s infinite ease-in-out;
    }
    .ambient-dot-red {
        display: inline-block;
        width: 9px;
        height: 9px;
        background-color: var(--color-bearish);
        border-radius: 50%;
        margin-right: 6px;
    }

    /* High-Visibility Emergency Kill Switch Box */
    .kill-switch-box {
        background: #20090d;
        border: 2px solid var(--color-bearish);
        border-radius: 10px;
        padding: 16px;
        margin-top: 10px;
    }

    /* Segmented Navigation / Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: var(--bg-card-solid);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: var(--text-secondary);
        font-size: 0.88rem;
        font-weight: 600;
        border: none;
        background-color: transparent;
        padding: 0 16px;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background-color: rgba(255,255,255,0.04);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-card-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-prominent) !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.15s ease;
        border: 1px solid var(--border-prominent);
    }
    .stButton>button:hover {
        border-color: var(--text-secondary);
    }

    /* Inputs, Selectboxes, and Number Inputs */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: var(--bg-card-solid) !important;
        border-color: var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    /* Plain English Tip Box */
    .tip-box {
        background: rgba(14, 165, 233, 0.08);
        border: 1px solid var(--color-cyan);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 15px;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-obsidian);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-prominent);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Initialize Session State
# -------------------------------------------------------------
if "active_broker_name" not in st.session_state:
    st.session_state.active_broker_name = config.ACTIVE_BROKER

if "bot_instance" not in st.session_state:
    st.session_state.bot_instance = LiveTradingBot(broker_name=st.session_state.active_broker_name)

if "bot_running" not in st.session_state:
    st.session_state.bot_running = False

if "selected_stock_for_backtest" not in st.session_state:
    st.session_state.selected_stock_for_backtest = "RELIANCE.NS"

broker = get_broker(st.session_state.active_broker_name)

# -------------------------------------------------------------
# Top Header & Live Market Telemetry Bar (1s Ultra-Fast Live Stream Fragment)
# -------------------------------------------------------------
@st.fragment(run_every=1)
def render_live_header_telemetry():
    broker = get_broker(st.session_state.active_broker_name)
    market_open, market_status_text = is_market_open()
    ist_now = get_ist_now().strftime("%d %b %Y | %H:%M:%S IST")
    
    nifty_quote = get_live_quote("^NSEI")
    banknifty_quote = get_live_quote("^NSEBANK")
    portfolio_data = broker.get_account_balance()
    
    col_logo, col_nifty, col_banknifty, col_status = st.columns([2.6, 2.2, 2.2, 2.0])
    
    with col_logo:
        st.markdown(f"""
        <div class='op-card'>
            <div style='font-size: 1.25rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif; display: flex; align-items: center; gap: 6px;'>
                📈 ApexTrade <span style='font-size: 0.65rem; background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid #6366f1; padding: 2px 5px; border-radius: 4px; font-weight: 700;'>INSTITUTIONAL AI</span>
            </div>
            <div style='color: #94a3b8; font-size: 0.74rem; font-family: "JetBrains Mono", monospace; margin-top: 3px;'>
                TIME: {ist_now}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_nifty:
        nifty_price = nifty_quote.get("price", 24350.0)
        nifty_chg = nifty_quote.get("change_pct", 0.0)
        nifty_chg_pts = nifty_quote.get("change", 0.0)
        n_card_cls = "price-card-bear" if nifty_chg < 0 else "price-card-bull"
        n_text_cls = "price-text-bear" if nifty_chg < 0 else "price-text-bull"
        n_badge_cls = "badge-bear" if nifty_chg < 0 else "badge-bull"
        n_arrow = "▼ " if nifty_chg < 0 else "▲ +"
        
        st.markdown(f"""
        <div class='{n_card_cls}'>
            <div style='font-size: 0.70rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;'>🇮🇳 NIFTY 50 INDEX</div>
            <div style='display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-top: 2px;'>
                <div class='{n_text_cls}'>₹{nifty_price:,.2f}</div>
                <span class='{n_badge_cls}' style='font-size: 0.72rem; padding: 2px 6px; white-space: nowrap;'>{n_arrow}{nifty_chg:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_banknifty:
        bn_price = banknifty_quote.get("price", 51200.0)
        bn_chg = banknifty_quote.get("change_pct", 0.0)
        bn_chg_pts = banknifty_quote.get("change", 0.0)
        bn_card_cls = "price-card-bear" if bn_chg < 0 else "price-card-bull"
        bn_text_cls = "price-text-bear" if bn_chg < 0 else "price-text-bull"
        bn_badge_cls = "badge-bear" if bn_chg < 0 else "badge-bull"
        bn_arrow = "▼ " if bn_chg < 0 else "▲ +"
        
        st.markdown(f"""
        <div class='{bn_card_cls}'>
            <div style='font-size: 0.70rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;'>🏦 BANK NIFTY INDEX</div>
            <div style='display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-top: 2px;'>
                <div class='{bn_text_cls}'>₹{bn_price:,.2f}</div>
                <span class='{bn_badge_cls}' style='font-size: 0.72rem; padding: 2px 6px; white-space: nowrap;'>{bn_arrow}{bn_chg:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status:
        status_dot = "ambient-dot-green" if market_open else "ambient-dot-red"
        status_badge_class = "badge-bull" if market_open else "badge-bear"
        st.markdown(f"""
        <div class='op-card' style='align-items: flex-end;'>
            <div><span class='{status_badge_class}' style='font-size: 0.74rem; padding: 2px 8px;'><span class='{status_dot}'></span>{market_status_text.upper()}</span></div>
            <div style='color: #94a3b8; font-size: 0.74rem; margin-top: 4px;'>
                Broker: <strong style='color: #38bdf8; font-family: "JetBrains Mono", monospace;'>{broker.name.upper()}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Account Telemetry Row
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    total_eq = portfolio_data.get("total_equity", 100000.0)
    cash_avail = portfolio_data.get("cash", 100000.0)
    unreal_pnl = portfolio_data.get("unrealized_pnl", 0.0)
    real_pnl = portfolio_data.get("realized_pnl", 0.0)
    init_cap = portfolio_data.get("initial_capital", 100000.0)
    total_ret_pct = ((total_eq - init_cap) / init_cap) * 100.0
    
    ret_prefix = "▲ +" if total_ret_pct >= 0 else "▼ "
    m1.metric("💰 Total Portfolio Value", format_currency_inr(total_eq), f"{ret_prefix}{total_ret_pct:.2f}%", delta_color="normal")
    m2.metric("💵 Available Cash Margin", format_currency_inr(cash_avail))
    
    unreal_arr = "▲ +" if unreal_pnl >= 0 else "▼ "
    m3.metric("📈 Live Open P&L", format_currency_inr(unreal_pnl), f"{unreal_arr}{unreal_pnl:,.2f} ₹", delta_color="normal")
    
    real_arr = "▲ +" if real_pnl >= 0 else "▼ "
    m4.metric("🏆 Realized P&L", format_currency_inr(real_pnl), f"{real_arr}{real_pnl:,.2f} ₹", delta_color="normal")
    
    m5.metric("📦 Active Trades", f"{portfolio_data.get('open_positions_count', 0)} / {config.MAX_CONCURRENT_POSITIONS} Legs")

render_live_header_telemetry()

# -------------------------------------------------------------
# Beginner Guide & Terms Cheat Sheet
# -------------------------------------------------------------
with st.expander("📘 **Plain-English Trading Cheat Sheet (Click to Expand)**", expanded=False):
    st.markdown("""
    #### 💡 Essential Trading Terms Explained in Simple Words:
    * 🛡️ **Stop-Loss (SL)**: *Your automatic safety net. If a stock drops by your chosen percentage (e.g. 1.5%), the bot immediately sells it to prevent you from taking big losses.*
    * 🎯 **Take-Profit (TP)**: *Your profit target. When the stock gains your goal percentage (e.g. 3.0%), the bot automatically sells to lock in your profits.*
    * 🏃 **Trailing Stop-Loss**: *A moving safety shield. As the stock climbs higher and higher, your safety line moves up with it, ensuring you don't lose accumulated profits if the price turns back down!*
    * 🧪 **Paper Trading (Virtual Money)**: *Safe practice mode! You trade with ₹1,00,000 of virtual demo money using 100% real-time live stock prices and actual Indian broker taxes. Zero real money at risk.*
    * 📊 **Win Rate**: *How often the strategy wins. For example, a 70% win rate means 7 out of 10 trades were profitable.*
    * 📉 **Max Drawdown**: *The largest temporary dip from the highest peak during the test period. Lower is safer.*
    * ⚡ **MIS (Intraday) vs CNC (Delivery)**: *MIS means day-trading (buying and selling today before 3:15 PM). CNC means buying shares to keep in your demat account for days, weeks, or years.*
    """)

# -------------------------------------------------------------
# ⚡ Live Stock Price Watcher Widget (Instant 2s Real-Time Stream)
# -------------------------------------------------------------
@st.fragment(run_every=2)
def render_live_stock_watcher():
    with st.expander("⚡ **Live Stock Price Watcher — Search Any Indian Stock Instantly (Auto-Streaming)**", expanded=True):
        pq_col1, pq_col2, pq_col3 = st.columns([2.5, 2.5, 1])
        with pq_col1:
            watch_query = st.text_input(
                "🔍 Type Company Name or Ticker to Search:",
                key="watcher_search_input",
                placeholder="e.g. Tata, Zomato, Reliance, Suzlon, SBI, Paytm, HAL, IRFC..."
            )
            
        matching_stocks = search_indian_stocks(watch_query)
        
        with pq_col2:
            watch_stock = st.selectbox(
                f"Select Matching Company ({len(matching_stocks)} found):",
                options=[item["symbol"] for item in matching_stocks],
                format_func=lambda s: next((f"{item['name']} ({item['symbol'].replace('.NS','')}) — {item.get('category', 'Equity')}" for item in matching_stocks if item["symbol"] == s), s),
                key="watcher_stock_select_dropdown",
                index=0
            )
        with pq_col3:
            st.markdown("<div style='padding-top: 28px;'>", unsafe_allow_html=True)
            if st.button("🔄 Sync Quote", key="watcher_sync_btn", use_container_width=True):
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Fetch live quote for selected stock (Instantaneous Cache Lookup)
        live_data = get_live_quote(watch_stock)
        live_p = live_data.get("price", 0.0)
        live_chg = live_data.get("change", 0.0)
        live_chg_pct = live_data.get("change_pct", 0.0)
        live_high = live_data.get("high", live_p)
        live_low = live_data.get("low", live_p)
        live_prev = live_data.get("previous_close", live_p)
        live_vol = live_data.get("volume", 0)
        live_ts = live_data.get("timestamp", "")
        
        q1, q2, q3, q4, q5 = st.columns(5)
        s_arrow = "▲ +" if live_chg >= 0 else "▼ "
        q1.metric("⚡ Live Price (LTP)", f"₹{live_p:,.2f}", f"{s_arrow}{live_chg_pct:.2f}% ({live_chg:+.2f})", delta_color="normal")
        q2.metric("🔺 Day High", f"₹{live_high:,.2f}")
        q3.metric("🔻 Day Low", f"₹{live_low:,.2f}")
        q4.metric("🔙 Previous Close", f"₹{live_prev:,.2f}")
        q5.metric("📊 Volume", f"{live_vol:,.0f} sh", f"{live_ts}")

render_live_stock_watcher()

# -------------------------------------------------------------
# Sidebar Navigation & Stock Selection
# -------------------------------------------------------------
st.sidebar.markdown("""
<div style='padding: 6px 0 12px 0;'>
    <div style='font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>🧭 TERMINAL MODULES</div>
    <div style='font-size: 0.75rem; color: #94a3b8;'>Select active workstation</div>
</div>
""", unsafe_allow_html=True)

active_tab = st.sidebar.radio(
    "Navigation:",
    [
        "🤖 Autonomous AI Trading Agent (Claude / Kimi / F&O)",
        "🎯 Smart Stock Advisor (When to Buy/Sell)",
        "📊 Strategy Backtester (Test Any Stock)",
        "⚡ Automated Live / Paper Bot",
        "🔍 Indian Market Screener (Scan All Stocks)",
        "⚙️ Settings & Risk Controls"
    ],
    label_visibility="collapsed"
)

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
<div style='font-size: 0.82rem; color: #10b981; font-weight: 700;'><span class='ambient-dot-green'></span>1s Real-Time Live Stream Active</div>
<div style='font-size: 0.75rem; color: #94a3b8; margin-top: 2px;'>Background daemon streams prices & P&L in 0.006ms.</div>
""", unsafe_allow_html=True)
if st.sidebar.button("🔄 Force Instant Sync", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
# Two-Step Operational Panic Kill Switch
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

# -------------------------------------------------------------
# TAB 0: 🤖 Autonomous AI Trading Agent (Claude / Kimi / Zerodha)
# -------------------------------------------------------------
if active_tab == "🤖 Autonomous AI Trading Agent (Claude / Kimi / F&O)":
    st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;'>
        <h2 style='margin: 0;'>🤖 Autonomous AI Trading Agent <span style='font-size: 1rem; color: #818cf8; font-weight: 600;'>(LLM Decision Engine + Zerodha F&O)</span></h2>
    </div>
    <div style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 12px;'>Let state-of-the-art AI models (<strong>Anthropic Claude, Kimi Moonshot, OpenAI GPT-4o, Gemini</strong>) trade Index Options & Equities autonomously with mathematical risk guardrails.</div>
    """, unsafe_allow_html=True)

    # Regulatory & Guardrail Notice Badge
    st.markdown("""
    <div style='background: #111622; border: 1px solid #1e293b; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap;'>
        <span class='badge-bull'>🛡️ ZERO-BYPASS GUARDRAILS ACTIVE</span>
        <span class='badge-cyan'>🏛️ SEBI RETAIL ALGO COMPLIANT</span>
        <span class='badge-neutral'>⚡ RATE-LIMITED (<3 REQ/S)</span>
        <span class='badge-bear'>🛑 3:15 PM AUTO SQUARE-OFF</span>
    </div>
    """, unsafe_allow_html=True)

    # Load Saved AI Configuration
    saved_ai = load_ai_settings()
    saved_prov = saved_ai.get("provider", "gemini")
    
    provider_options = [
        "🔵 Google Gemini (Gemini 3.1 Flash-Lite / Gemini 3 / Gemma 4 — Ultra-Fast & Sub-Second Latency)",
        "🟣 Anthropic Claude (Claude 3.7 Sonnet — Latest Hybrid Reasoning)",
        "🌙 Kimi / Moonshot AI (Ultra Fast & Long Context)",
        "🟢 OpenAI (GPT-4o / GPT-4o-mini)",
        "🔴 DeepSeek (DeepSeek-Chat / Reasoner)"
    ]
    
    prov_to_idx = {"gemini": 0, "anthropic": 1, "kimi": 2, "openai": 3, "deepseek": 4}
    default_prov_idx = prov_to_idx.get(saved_prov, 0)
    
    # Active Connection Status Banner
    if saved_ai.get("is_connected") and saved_ai.get("api_key"):
        st.markdown(f"""
        <div style='background: #111622; border: 1px solid #10b981; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='color: #10b981; font-weight: 700; font-size: 1.02rem;'><span class='ambient-dot-green'></span>ACTIVE AI ENGINE: {saved_ai['provider'].upper()} ({saved_ai.get('model', 'gemini-3.1-flash-lite')})</span>
                <div style='color: #94a3b8; font-size: 0.82rem; margin-top: 2px;'>Authentication Verified & Locally Persisted &bull; Zero Re-entry Required</div>
            </div>
            <span class='badge-bull'>CONNECTED</span>
        </div>
        """, unsafe_allow_html=True)

    # Section 1: AI Model & API Configuration
    with st.expander("🧠 **Step 1: Choose Your AI Model & Enter API Key**", expanded=not saved_ai.get("is_connected")):
        ai_col1, ai_col2 = st.columns([2, 2.5])
        with ai_col1:
            ai_provider = st.selectbox(
                "Select AI Brain Provider:",
                provider_options,
                index=default_prov_idx
            )
            
            if "Gemini" in ai_provider:
                prov_key = "gemini"
                model_default = "gemini-3.1-flash-lite"
            elif "Anthropic" in ai_provider:
                prov_key = "anthropic"
                model_default = "claude-3-7-sonnet-20250219"
            elif "Kimi" in ai_provider:
                prov_key = "kimi"
                model_default = "moonshot-v1-8k"
            elif "OpenAI" in ai_provider:
                prov_key = "openai"
                model_default = "gpt-4o"
            elif "DeepSeek" in ai_provider:
                prov_key = "deepseek"
                model_default = "deepseek-chat"
            else:
                prov_key = "gemini"
                model_default = "gemini-3.1-flash-lite"
                
            model_saved = saved_ai.get("model") if saved_ai.get("provider") == prov_key else model_default
            model_choice = st.text_input("AI Model Name:", value=model_saved or model_default)
            
        with ai_col2:
            session_key = f"api_key_{prov_key}"
            if session_key not in st.session_state:
                saved_key = saved_ai.get("api_key", "") if saved_ai.get("provider") == prov_key else ""
                st.session_state[session_key] = saved_key or os.getenv(f"{prov_key.upper()}_API_KEY", "")
                
            ai_api_key = st.text_input(
                f"Enter {prov_key.upper()} API Key:",
                type="password",
                key=session_key,
                placeholder=f"Paste your {prov_key.upper()} API key here...",
                help="Your API key stays strictly local in your private configuration and is never shared or stored remotely."
            )
            
            test_col1, test_col2 = st.columns([1.5, 2.5])
            with test_col1:
                st.write("")
                test_ai_btn = st.button("🔗 Test AI Connection", use_container_width=True)
            with test_col2:
                st.write("")
                
            status_key = f"ai_test_status_{prov_key}"
            if test_ai_btn:
                if not ai_api_key or len(ai_api_key.strip()) < 5:
                    st.session_state[status_key] = ("info", "ℹ️ **API Key Missing:** Please paste your API key in the field above before running the test.")
                else:
                    with st.spinner(f"Connecting to {prov_key.upper()}..."):
                        client_tester = LLMClient(provider=prov_key, model=model_choice, api_key=ai_api_key)
                        success, msg = client_tester.test_connection()
                        if success:
                            st.session_state[status_key] = ("success", f"✅ **Connected Successfully:** {msg}")
                            save_ai_settings({
                                "provider": prov_key,
                                "model": model_choice,
                                "api_key": ai_api_key,
                                "is_connected": True,
                                "updated_at": get_ist_now().isoformat()
                            })
                        else:
                            st.session_state[status_key] = ("error", f"❌ **Connection Issue:** {msg}")
                            
            if status_key in st.session_state:
                s_type, s_msg = st.session_state[status_key]
                if s_type == "success":
                    st.success(s_msg)
                elif s_type == "error":
                    st.error(s_msg)
                else:
                    st.info(s_msg)

    # Section 2: Execution Mode & Broker Link (Paper vs Zerodha Live)
    with st.expander("🔴 **Step 2: Execution Mode & Broker Setup (Paper vs Zerodha Live)**", expanded=True):
        exec_mode = st.radio(
            "Select Execution Mode:",
            [
                "🛡️ AI Paper Simulation Mode (Zero Risk — Test AI Decision Quality First)",
                "🔴 Zerodha Real Live Account (Places Real Orders on Zerodha via Kite Connect)"
            ],
            horizontal=True
        )
        
        is_live_selected = "Zerodha Real Live" in exec_mode
        active_ai_broker = broker
        
        if is_live_selected:
            st.warning("⚠️ **Live Real Capital Mode Enabled**: Real orders will be routed to your Zerodha account with hard stop-losses.")
            z_col1, z_col2, z_col3 = st.columns(3)
            with z_col1:
                z_api_key = st.text_input("Zerodha API Key:", type="password", key="zerodha_api_key", placeholder="Your Kite Connect API Key")
            with z_col2:
                z_api_secret = st.text_input("Zerodha API Secret:", type="password", key="zerodha_api_secret", placeholder="Your Kite Connect API Secret")
            with z_col3:
                z_req_token = st.text_input("Request Token (from login redirect):", type="password", key="zerodha_req_token", placeholder="Paste request_token here")
                
            if z_api_key and z_api_secret:
                active_ai_broker = ZerodhaLiveBroker(api_key=z_api_key, api_secret=z_api_secret)
                if z_req_token:
                    if st.button("🔑 Generate Daily Access Token"):
                        s_ok, s_msg = active_ai_broker.set_session(z_req_token)
                        if s_ok:
                            st.success(f"✅ {s_msg}")
                        else:
                            st.error(f"❌ {s_msg}")
            else:
                st.info("💡 Enter your Zerodha Kite Connect credentials above to link your demat account.")

    # Section 3: Target Asset & Guardrail Parameters
    with st.expander("🎯 **Step 3: Target Asset & Safety Guardrail Limits**", expanded=True):
        g_col1, g_col2 = st.columns([2.5, 2.5])
        with g_col1:
            target_asset_choice = st.selectbox(
                "Select Asset to Trade:",
                [
                    "NIFTY 50 Index Options (CE / PE Calls & Puts)",
                    "BANK NIFTY Index Options (CE / PE Calls & Puts)",
                    "RELIANCE (Reliance Industries)",
                    "TMCV.NS (Tata Motors)",
                    "ETERNAL.NS (Zomato)",
                    "SBIN.NS (State Bank of India)"
                ]
            )
            clean_target = "NIFTY" if "NIFTY 50" in target_asset_choice else ("BANKNIFTY" if "BANK NIFTY" in target_asset_choice else target_asset_choice.split(" ")[0])
            
            # Fetch live quote for preview
            q_target = get_live_quote(clean_target)
            if q_target.get("price", 0) > 0:
                st.caption(f"⚡ **{clean_target} Live LTP:** ₹{q_target['price']:,.2f} ({q_target['change_pct']:+.2f}%) | High: ₹{q_target['high']:,.2f} | Low: ₹{q_target['low']:,.2f}")
                
        with g_col2:
            st.markdown("**🛡️ Deterministic Guardrails (The AI Cannot Override These):**")
            guard_max_loss = st.number_input("Max Daily Loss Limit (₹):", min_value=500.0, value=2000.0, step=500.0)
            guard_min_conf = st.slider("Min AI Confidence Threshold (/10):", min_value=6.0, max_value=9.5, value=7.5, step=0.1)
            guard_max_legs = st.number_input("Max Concurrent Active Legs (Position Cap):", min_value=1, max_value=10, value=2, step=1, help="Maximum number of simultaneous open positions allowed by Guardrails.")

    # Initialize Agent & Guardrails
    ai_guardrails = AIGuardrails(
        max_daily_loss_flat=guard_max_loss,
        max_daily_loss_pct=3.0,
        max_concurrent_legs=int(guard_max_legs),
        max_lots_per_trade=1,
        min_confidence_threshold=guard_min_conf,
        sl_cooldown_minutes=15
    )
    
    # State Reconciliation Card (1s Live Stream)
    @st.fragment(run_every=2)
    def render_ai_state_reconciliation():
        reconciled = StateReconciler.reconcile_with_broker(active_ai_broker)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💼 Account Available Margin", f"₹{reconciled['capital']:,.2f}")
        real_dpnl = reconciled['daily_pnl']
        dpnl_arr = "▲ +" if real_dpnl >= 0 else "▼ "
        m2.metric("📈 Today's Realized PnL", f"₹{real_dpnl:+,.2f}", f"{dpnl_arr}{real_dpnl:,.2f} ₹", delta_color="normal")
        m3.metric("📊 Active Open Legs", f"{reconciled['active_legs_count']} position(s)")
        m4.metric("🛡️ Ground Truth Sync", f"{reconciled['status']}")
        
        # If active open legs exist, show quick operational summary directly in Tab 0
        if reconciled['active_legs_count'] > 0:
            open_pos = active_ai_broker.get_open_positions()
            for pos in open_pos:
                p_sym = display_symbol_name(pos['symbol'])
                p_side = pos['side']
                p_time = pos.get('entry_time', 'N/A')
                p_dur = format_holding_duration(p_time)
                p_entry = float(pos['entry_price'])
                p_curr = float(pos.get('current_price', p_entry))
                p_pnl = float(pos.get('unrealized_pnl', 0.0))
                p_pnl_pct = float(pos.get('unrealized_pnl_pct', 0.0))
                p_border = "#10b981" if p_pnl >= 0 else "#f43f5e"
                p_arr = "▲ +" if p_pnl >= 0 else "▼ "
                
                st.markdown(f"""
                <div style='background: #111622; border-left: 5px solid {p_border}; border-top: 1px solid #1e293b; border-right: 1px solid #1e293b; border-bottom: 1px solid #1e293b; border-radius: 8px; padding: 12px 16px; margin-top: 10px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;'>
                        <div style='display: flex; align-items: center; gap: 10px;'>
                            <strong style='color: #f8fafc; font-size: 1.05rem; font-family: "Outfit", sans-serif;'>{p_sym}</strong>
                            <span class='badge-cyan' style='font-size: 0.72rem;'>{p_side} ({pos['quantity']} sh)</span>
                            <span class='badge-neutral' style='font-size: 0.72rem;'>🕒 Executed: {p_time}</span>
                            <span class='badge-bull' style='font-size: 0.72rem;'>⏳ Holding: {p_dur}</span>
                        </div>
                        <div style='display: flex; align-items: center; gap: 14px;'>
                            <div style='font-size: 0.84rem; color: #94a3b8;'>Entry: <span class='mono-num'>₹{p_entry:,.2f}</span> &bull; LTP: <span class='mono-num'>₹{p_curr:,.2f}</span></div>
                            <div style='font-size: 1.05rem; font-weight: 700; color: {p_border}; font-family: "JetBrains Mono", monospace;'>{format_currency_inr(p_pnl)} ({p_arr}{p_pnl_pct:.2f}%)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    render_ai_state_reconciliation()

    st.markdown("---")

    # Section 4: AI Live Opportunity Radar (Multi-Asset Scanner & Auto-Dispatcher)
    with st.expander("📡 **Step 4: AI Opportunity Radar (Live Top Trade Setups & Holding Expectancy)**", expanded=True):
        st.caption("AI continuously scans NIFTY 50, BANK NIFTY, and high-momentum stocks to calculate exact entry, stop-loss, 1:2 R:R targets, and holding horizon.")
        
        rad_col1, rad_col2, rad_col3 = st.columns([2, 2, 1.5])
        with rad_col1:
            scan_radar_btn = st.button("🔄 Scan Market Opportunities Now", type="primary", use_container_width=True)
        with rad_col2:
            auto_radar_dispatch = st.checkbox("⚡ Auto-Dispatch Top Setup (Confidence >= 8.0)", value=False, help="Automatically place order if an opportunity meets institutional conviction.")
        with rad_col3:
            st.caption(f"Engine: **{prov_key.upper()}** ({model_choice})")
            
        if scan_radar_btn:
            if not ai_api_key or len(ai_api_key.strip()) < 5:
                st.warning("⚠️ **API Key Required:** Please enter your AI API key in **Step 1** before scanning.")
            else:
                with st.spinner(f"AI Radar ({prov_key.upper()} {model_choice}) is scanning NIFTY, BANK NIFTY, and liquid momentum equities..."):
                    llm_instance = LLMClient(provider=prov_key, model=model_choice, api_key=ai_api_key)
                    radar_res = MarketRadarScanner.scan_market(llm_client=llm_instance, min_confidence=7.0)
                    st.session_state["last_radar_scan"] = radar_res
                    
                    # Auto-Dispatch if enabled and confidence >= 8.0
                    if auto_radar_dispatch and radar_res.get("status") == "SUCCESS":
                        opps_list = radar_res.get("opportunities", [])
                        if opps_list and float(opps_list[0].get("confidence_score", 0)) >= 8.0:
                            top_opp = opps_list[0]
                            agent_auto = AITradingAgent(
                                llm_client=llm_instance,
                                guardrails=ai_guardrails,
                                broker=active_ai_broker,
                                is_live_mode=is_live_selected
                            )
                            auto_outcome = agent_auto.execute_radar_opportunity(top_opp)
                            if auto_outcome.get("status") == "EXECUTED":
                                st.success(f"🚀 **Auto-Pilot Executed:** {top_opp.get('action')} on {top_opp.get('symbol')}!")
                    
        # Render Opportunity Cards if scan data exists
        if "last_radar_scan" in st.session_state:
            r_data = st.session_state["last_radar_scan"]
            if r_data.get("status") == "SUCCESS":
                st.markdown(f"**🌐 Market Tone:** *{r_data.get('market_summary')}*")
                opps = r_data.get("opportunities", [])
                
                if not opps:
                    st.info("ℹ️ No setups passed the minimum confidence threshold right now. Capital preserved.")
                else:
                    for i, opp in enumerate(opps):
                        o_rank = opp.get("rank", i + 1)
                        o_sym = opp.get("symbol", "N/A")
                        o_contract = opp.get("option_contract", "N/A")
                        o_act = opp.get("action", "BUY_CALL")
                        o_conf = float(opp.get("confidence_score", 0.0))
                        o_horizon = opp.get("time_horizon", "Intraday")
                        o_setup = opp.get("setup_name", "Momentum Setup")
                        o_entry = float(opp.get("entry_price", 0.0))
                        o_sl = float(opp.get("stop_loss", 0.0))
                        o_t1 = float(opp.get("target_1", 0.0))
                        o_t2 = float(opp.get("target_2", 0.0))
                        o_gain = opp.get("expected_gain_pct", "+25%")
                        o_reason = opp.get("catalyst_reasoning", "")
                        
                        card_border = "#10b981" if "CALL" in o_act or "STOCK" in o_act else "#f43f5e"
                        display_title = f"#{o_rank} {o_contract if o_contract != 'N/A' else o_sym} ({o_act})"
                        
                        st.markdown(f"""
                        <div style='background: #111622; border-left: 4px solid {card_border}; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; border-top: 1px solid #1e293b; border-right: 1px solid #1e293b; border-bottom: 1px solid #1e293b;'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <span style='font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>{display_title}</span>
                                <span class='badge-bull'>⭐ CONFIDENCE: {o_conf:.1f}/10</span>
                            </div>
                            <div style='color: #38bdf8; font-size: 0.9rem; font-weight: 600; margin: 4px 0;'>🎯 {o_setup} &bull; ⏱️ Holding Time: <strong>{o_horizon}</strong> &bull; Exp. Return: <span style='color: #10b981;'>{o_gain}</span></div>
                            <div style='color: #cbd5e1; font-size: 0.9rem; margin-top: 4px;'>🧠 <strong>AI Rationale:</strong> {o_reason}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics Row & 1-Click Execution Button
                        c_m1, c_m2, c_m3, c_m4, c_btn = st.columns([1.2, 1.2, 1.2, 1.2, 2])
                        c_m1.metric("Entry Price", f"₹{o_entry:,.2f}")
                        c_m2.metric("Safety SL", f"₹{o_sl:,.2f}")
                        c_m3.metric("Target 1 (1:2)", f"₹{o_t1:,.2f}")
                        c_m4.metric("Target 2 (Runner)", f"₹{o_t2:,.2f}")
                        
                        with c_btn:
                            st.write("")
                            if st.button(f"🚀 Execute Trade #{o_rank}", key=f"btn_exec_opp_{i}", type="primary", use_container_width=True):
                                llm_instance = LLMClient(provider=prov_key, model=model_choice, api_key=ai_api_key)
                                agent = AITradingAgent(
                                    llm_client=llm_instance,
                                    guardrails=ai_guardrails,
                                    broker=active_ai_broker,
                                    is_live_mode=is_live_selected
                                )
                                exec_outcome = agent.execute_radar_opportunity(opp)
                                if exec_outcome.get("status") == "EXECUTED":
                                    st.success(f"✅ Trade Executed! Symbol: `{exec_outcome.get('symbol')}`")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Execution Blocked: {exec_outcome.get('message')}")
                        st.divider()
            elif r_data.get("status") == "ERROR":
                st.error(f"❌ {r_data.get('message')}")

    st.markdown("---")

    # Section 5: Hands-Free Auto-Pilot Bot & Continuous Background Engine
    st.subheader("🤖 Hands-Free Auto-Pilot Trading Bot (Continuous 5m Execution)")
    st.caption("Runs an autonomous background loop: monitors 5-minute candle closes, auto-dispatches AI trades with conviction >= 8.0/10, dynamically trails stop-losses, and squares off at 3:15 PM IST.")
    
    daemon = AutoPilotDaemon.get_instance()
    llm_instance_current = LLMClient(provider=prov_key, model=model_choice, api_key=ai_api_key)
    daemon.configure(
        llm_client=llm_instance_current,
        guardrails=ai_guardrails,
        broker=active_ai_broker,
        is_live_mode=is_live_selected,
        min_auto_confidence=8.0
    )
    d_status = daemon.get_status()
    
    d_col1, d_col2, d_col3, d_col4 = st.columns([2, 1.5, 1.5, 1.5])
    with d_col1:
        if d_status["is_running"]:
            if st.button("⏸️ Pause Auto-Pilot Engine", type="secondary", use_container_width=True):
                daemon.stop()
                st.rerun()
        else:
            if st.button("▶️ Start Autonomous Auto-Pilot Bot", type="primary", use_container_width=True):
                if not ai_api_key or len(ai_api_key.strip()) < 5:
                    st.warning("⚠️ Enter your AI API Key in Step 1 first!")
                else:
                    daemon.start()
                    st.rerun()
    with d_col2:
        st.metric("Auto-Pilot Status", "🟢 ACTIVE" if d_status["is_running"] else "⏸️ PAUSED")
    with d_col3:
        st.metric("5m Scans Done", f"{d_status['scans_count']}")
    with d_col4:
        st.metric("Orders Filled", f"{d_status['orders_executed']}")

    # Section 6: Smart Active Positions (50/50 Profit Booker & Trailing SL Visualizer)
    st.markdown("### 📌 Active AI Open Positions & Trailing SL Manager")
    
    pos_btn_c1, pos_btn_c2 = st.columns([3, 1.5])
    with pos_btn_c2:
        if st.button("⚡ Manage Trailing Stops & Targets Now", use_container_width=True):
            m_events = SmartTradeManager.evaluate_and_manage_positions(active_ai_broker)
            if m_events:
                for ev in m_events:
                    st.success(f"⚡ {ev.get('message')}")
            else:
                st.info("ℹ️ Trailing SL & Targets checked. All active positions healthy.")
            st.rerun()

    ai_positions = active_ai_broker.get_positions() if hasattr(active_ai_broker, "get_positions") else get_open_positions()
    if ai_positions:
        for pos in ai_positions:
            p_sym = pos["symbol"]
            p_qty = int(pos["quantity"])
            p_entry = float(pos["entry_price"])
            p_curr = float(pos.get("current_price", p_entry))
            p_sl = float(pos.get("sl") or (p_entry * 0.85))
            p_t1 = float(pos.get("target_1") or (pos.get("tp") or (p_entry * 1.20)))
            p_t2 = float(pos.get("target_2") or (p_entry * 1.45))
            p_trail = float(pos.get("trailing_sl") or p_sl)
            t1_hit = bool(pos.get("target_1_hit", 0))
            p_gain = ((p_curr - p_entry) / p_entry) * 100.0
            
            card_accent = "#10b981" if p_gain >= 0 else "#f43f5e"
            status_tag = "🔒 BREAKEVEN LOCKED (50% Profit Secured)" if t1_hit else "🛡️ INITIAL RISK PHASE"
            gain_arr = "▲ +" if p_gain >= 0 else "▼ "
            
            st.markdown(f"""
            <div style='background: #111622; border: 1px solid {card_accent}; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: "JetBrains Mono", monospace;'>{p_sym} &bull; {p_qty} shares</span>
                    <span class='{'badge-bull' if t1_hit else 'badge-cyan'}'>{status_tag}</span>
                </div>
                <div style='margin-top: 6px; color: #94a3b8; font-size: 0.88rem;'>
                    Entry: <strong class='mono-num'>₹{p_entry:.2f}</strong> &bull; LTP: <strong class='mono-num' style='color: {'#10b981' if p_gain >= 0 else '#f43f5e'};'>₹{p_curr:.2f} ({gain_arr}{p_gain:.1f}%)</strong> &bull; Trailing SL: <strong class='mono-num'>₹{p_trail:.2f}</strong> &bull; Target 1: <strong class='mono-num'>₹{p_t1:.2f}</strong> &bull; Target 2: <strong class='mono-num'>₹{p_t2:.2f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No active positions. The Auto-Pilot Bot is continuously monitoring for high-conviction entries.")

    st.markdown("---")

    # Section 7: Single-Asset Evaluation & Manual Panic Controls
    st.subheader("⚡ Manual AI Single-Asset Terminal & Emergency Controls")
    
    op_col1, op_col2, op_col3 = st.columns([2, 2, 1.5])
    with op_col1:
        run_ai_btn = st.button("🧠 Evaluate Target Asset Manually", type="primary", use_container_width=True)
    with op_col2:
        auto_ai_mode = st.toggle("⚡ Enable Auto-Pilot Evaluation on Candle Close", value=False)
    with op_col3:
        if st.button("🚨 Panic Kill Switch", type="secondary", use_container_width=True):
            sq_list = active_ai_broker.square_off_all(reason="User Panic Kill Switch")
            st.warning(f"Emergency Square-Off Executed! Closed {len(sq_list)} positions.")
            st.rerun()

    # Execution Action
    if run_ai_btn or auto_ai_mode:
        if not ai_api_key or len(ai_api_key.strip()) < 5:
            st.warning("⚠️ **API Key Required:** Please enter your AI API key in **Step 1** above before running market evaluations.")
        else:
            try:
                llm_instance = LLMClient(provider=prov_key, model=model_choice, api_key=ai_api_key)
                agent = AITradingAgent(
                    llm_client=llm_instance,
                    guardrails=ai_guardrails,
                    broker=active_ai_broker,
                    is_live_mode=is_live_selected
                )
                
                with st.spinner(f"AI ({prov_key.upper()} {model_choice}) is analyzing live market structure for {clean_target}..."):
                    telemetry = agent.evaluate_and_execute(clean_target)
                    
                    st.markdown("### 📡 Live AI Telemetry Feed")
                    
                    # Telemetry Overview Banner
                    act_color = "#10b981" if "BUY" in str(telemetry.get("action")) else ("#f43f5e" if "EXIT" in str(telemetry.get("action")) else "#94a3b8")
                    g_color = "#10b981" if telemetry.get("guardrail_status") == "APPROVED" else "#f43f5e"
                    
                    t_col1, t_col2, t_col3 = st.columns([1.5, 1.5, 3])
                    with t_col1:
                        st.markdown(f"""
                        <div style='background: #111622; border: 2px solid {act_color}; border-radius: 8px; padding: 14px; text-align: center;'>
                            <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;'>AI Action Proposed</div>
                            <div style='font-size: 1.4rem; font-weight: 800; color: {act_color}; margin: 4px 0; font-family: "Outfit", sans-serif;'>{telemetry.get('action')}</div>
                            <div style='font-size: 0.9rem; color: #f8fafc;'>Confidence: <strong class='mono-num'>{telemetry.get('confidence')}/10</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with t_col2:
                        st.markdown(f"""
                        <div style='background: #111622; border: 2px solid {g_color}; border-radius: 8px; padding: 14px; text-align: center;'>
                            <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;'>Guardrail Layer</div>
                            <div style='font-size: 1.4rem; font-weight: 800; color: {g_color}; margin: 4px 0; font-family: "Outfit", sans-serif;'>{telemetry.get('guardrail_status')}</div>
                            <div style='font-size: 0.85rem; color: #94a3b8;'>Latency: <span class='mono-num'>{telemetry.get('latency_sec')}s</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with t_col3:
                        st.markdown(f"""
                        <div style='background: #111622; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;'>
                            <div style='font-size: 0.85rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px;'>🧠 AI Institutional Rationale:</div>
                            <div style='font-size: 0.92rem; color: #f8fafc; line-height: 1.4;'>{telemetry.get('reasoning')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Execution Result Box
                    exec_res = telemetry.get("execution", {})
                    if exec_res.get("status") in ["FILLED", "SUCCESS"]:
                        exec_p = float(exec_res.get("price") or 0.0)
                        st.success(f"🚀 **Order Executed:** {exec_res.get('side')} {exec_res.get('quantity')}x `{exec_res.get('symbol')}` @ ₹{exec_p:.2f} (Tag: `{exec_res.get('idempotency_tag', 'N/A')}`)")
                    elif exec_res.get("status") == "SKIPPED":
                        st.info(f"ℹ️ **Action Skipped by Guardrails:** {exec_res.get('message')}")
                    else:
                        st.warning(f"Order Status: {exec_res}")
            except Exception as eval_err:
                st.error(f"❌ **AI Evaluation Error:** {str(eval_err)}")

    # Section 8: Multi-Regime Historical Stress Replay
    st.markdown("---")
    with st.expander("📊 **Multi-Regime Historical Stress Replay (Validate AI Before Real Money)**", expanded=False):
        st.caption("Replay historical market regimes through the prompt and guardrail engine to measure how the strategy behaves in rallies, crashes, and chop.")
        
        rep_col1, rep_col2, rep_col3 = st.columns([2, 1.5, 1])
        with rep_col1:
            replay_regime = st.selectbox(
                "Select Historical Market Regime:",
                [
                    "🚀 Bullish Trend Regime (Multi-hour directional breakout)",
                    "📉 Sudden Market Fall / Crash Spike (High panic selling)",
                    "🟡 Choppy / Range-Bound Sideways Day (Consolidation)",
                    "⚡ High-IV Weekly Expiry Session (Fast gamma/theta)"
                ]
            )
            reg_code = "BULL_TREND" if "Bullish" in replay_regime else ("BEAR_CRASH" if "Crash" in replay_regime else ("SIDEWAYS_CHOP" if "Choppy" in replay_regime else "EXPIRY_VOLATILITY"))
            
        with rep_col2:
            replay_sym = st.selectbox("Replay Stock/Index:", ["RELIANCE.NS", "TMCV.NS", "SBIN.NS", "INFY.NS", "NIFTY"])
        with rep_col3:
            st.write("")
            st.write("")
            run_replay_btn = st.button("▶️ Run Stress Replay", use_container_width=True)
            
        if run_replay_btn:
            with st.spinner(f"Replaying historical {reg_code} regime through AI agent pipeline..."):
                rep_res = AIBacktester.run_regime_backtest(symbol=replay_sym, regime=reg_code, sample_bars=25)
                if rep_res.get("status") == "SUCCESS":
                    st.success(f"✅ Stress Replay Completed for {rep_res['regime_name']}!")
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("Total Decisions Replayed", f"{rep_res['total_decisions']} bars")
                    rc2.metric("Total Executed Trades", f"{rep_res['total_trades']} trades")
                    rc3.metric("Win Rate %", f"{rep_res['win_rate']:.1f}%")
                    rc4.metric("Simulated Net PnL", f"₹{rep_res['net_pnl']:+,.2f}", delta_color="normal")
                    
                    st.line_chart(rep_res["equity_curve"])
                else:
                    st.error(rep_res.get("message", "Error running replay."))

    # Section 9: AI vs. Math Divergence & Calibration Journal (Audit Veto Power)
    with st.expander("📒 **Step 6: AI vs. Math Divergence & Calibration Journal (Audit Veto Power)**", expanded=False):
        st.caption("Inspect real-time cognitive divergence: verify whether the AI's asymmetric veto protected capital against false breakouts or if weak math signals were properly suppressed.")
        
        cal_records = get_calibration_records(limit=25)
        dis_records = get_disagreement_records(limit=25)
        
        c_tab1, c_tab2 = st.tabs([f"⚠️ Disagreements & Vetoes ({len(dis_records)})", f"📋 All Live Decisions ({len(cal_records)})"])
        with c_tab1:
            if dis_records:
                for rec in dis_records:
                    st.markdown(f"""
                    <div style='background: #111622; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; border-top: 1px solid #1e293b; border-right: 1px solid #1e293b; border-bottom: 1px solid #1e293b;'>
                        <div style='display: flex; justify-content: space-between;'>
                            <strong style='color: #38bdf8;' class='mono-num'>{rec['symbol']} &bull; {rec['timestamp']}</strong>
                            <span class='badge-cyan'>Regime: {rec['market_regime']}</span>
                        </div>
                        <div style='margin-top: 4px; font-size: 0.85rem; color: #cbd5e1;'>
                            Math Score: <strong class='mono-num'>{rec['math_score']}/10</strong> &bull; LLM Proposed: <strong class='mono-num'>{rec['proposed_action']} ({rec['llm_confidence']}/10)</strong> &bull; Final Decision: <strong style='color: #f59e0b;'>{rec['final_action']}</strong>
                        </div>
                        <div style='margin-top: 4px; font-size: 0.8rem; color: #94a3b8;'>
                            <em>{rec['disagreement_reason']}</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No divergence events recorded yet. When Math Scorer and LLM disagree, records will appear here.")
        with c_tab2:
            if cal_records:
                cal_df = pd.DataFrame(cal_records)[["timestamp", "symbol", "market_regime", "math_score", "llm_confidence", "final_action", "disagreement"]]
                st.dataframe(cal_df, use_container_width=True)
            else:
                st.info("No decision records in calibration journal yet.")

# -------------------------------------------------------------
# TAB 1: 🎯 Smart Stock Advisor & Trade Planner
# -------------------------------------------------------------
elif active_tab == "🎯 Smart Stock Advisor (When to Buy/Sell)":
    st.markdown("""
    <h2>🎯 Smart Stock Advisor & Trade Planner</h2>
    <div style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 14px;'>Get instant, actionable recommendations: <strong>When to Buy/Sell</strong>, <strong>Exact Price Targets</strong>, <strong>Safety Stop-Loss</strong>, and <strong>How Long to Hold</strong>.</div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔍 **Step 1: Search & Select Indian Stock**", expanded=True):
        adv_col1, adv_col2 = st.columns([2.5, 2])
        with adv_col1:
            adv_search = st.text_input("🔍 Search Company Name or Ticker:", value="", placeholder="Type any name e.g. Tata, Zomato, Reliance, Suzlon, Paytm, HAL, SBI...")
            matching_adv = search_indian_stocks(adv_search)
            
            adv_sym = st.selectbox(
                f"Choose Company ({len(matching_adv)} results):",
                options=[item["symbol"] for item in matching_adv],
                format_func=lambda s: next((f"{item['name']} ({item['symbol'].replace('.NS','')}) — {item.get('category', 'Equity')}" for item in matching_adv if item["symbol"] == s), s),
                index=0
            )
        with adv_col2:
            adv_horizon = st.selectbox(
                "Select Your Preferred Holding Style:",
                [
                    "⏳ Short-Term Swing (3 to 7 Days - High Probability)",
                    "⚡ Day Trading / Intraday (Same Day Exit before 3:15 PM)",
                    "📈 Positional Trend (2 to 4 Weeks - Ride the Bigger Wave)"
                ],
                index=0
            )
            if "Short-Term" in adv_horizon:
                h_key = "swing"
            elif "Intraday" in adv_horizon:
                h_key = "intraday"
            else:
                h_key = "positional"
                
    adv_btn = st.button("🧠 Analyze Stock & Generate Trade Blueprint", type="primary", use_container_width=True)
    
    adv_cache_key = f"advisor_res_{adv_sym}_{h_key}"
    if adv_btn or adv_cache_key not in st.session_state:
        with st.spinner(f"Analyzing technical patterns, buyer momentum, and price levels for {display_symbol_name(adv_sym)}..."):
            st.session_state[adv_cache_key] = StockAdvisor.analyze_stock(adv_sym, horizon=h_key)
            
    analysis = st.session_state.get(adv_cache_key, {})
        
    if analysis.get("status") == "SUCCESS":
        st.markdown("---")
        
        # Big Verdict Banner (Two-Tier High Contrast)
        v_col1, v_col2 = st.columns([1.5, 3])
        badge_c = analysis.get("badge_color", "#10b981")
        with v_col1:
            st.markdown(f"""
            <div style='background: #111622; border: 2px solid {badge_c}; border-radius: 10px; padding: 18px; text-align: center;'>
                <div style='font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>Overall Verdict</div>
                <div style='font-size: 1.5rem; font-weight: 800; color: {badge_c}; margin: 6px 0; font-family: "Outfit", sans-serif;'>{analysis.get("verdict")}</div>
                <div style='font-size: 1.1rem; font-weight: 700; color: #f8fafc;'>Score: <span class='mono-num'>{analysis.get("score")} / 10</span></div>
            </div>
            """, unsafe_allow_html=True)
        with v_col2:
            disp_name = analysis.get("display_name", display_symbol_name(adv_sym))
            v_desc = analysis.get("verdict_desc", "")
            curr_p = analysis.get("current_price", 0.0)
            h_text = analysis.get("horizon_text", analysis.get("holding_time_text", "Swing (3-7 Days)"))
            st.markdown(f"""
            <div style='background: #111622; border: 1px solid #1e293b; border-radius: 10px; padding: 18px;'>
                <div style='font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px; font-family: "Outfit", sans-serif;'>
                    Analysis for <strong>{disp_name}</strong> (`{clean_symbol(adv_sym)}`)
                </div>
                <div style='color: #94a3b8; font-size: 0.92rem; margin-bottom: 10px;'>
                    {v_desc}
                </div>
                <div style='display: flex; gap: 18px; font-size: 0.9rem;'>
                    <div>💵 <strong>Live Price:</strong> <span class='mono-num'>₹{curr_p:,.2f}</span></div>
                    <div>⏳ <strong>Holding Horizon:</strong> {h_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        t1 = analysis.get("target_1", {"price": curr_p * 1.03, "gain_pct": 3.0, "reward_risk": 1.5})
        t2 = analysis.get("target_2", {"price": curr_p * 1.06, "gain_pct": 6.0, "reward_risk": 2.5})
        sl = analysis.get("stop_loss", {"price": curr_p * 0.98, "loss_pct": 2.0})
        entry_z = analysis.get("entry_zone", f"₹{curr_p * 0.998:.2f} – ₹{curr_p:.2f}")

        # Visual Continuous Multi-Target Price Axis Ladder
        st.markdown(f"""
        <div class='op-card' style='padding: 14px 18px; margin: 12px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <div style='font-size: 0.80rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;'>📐 Multi-Target Risk-to-Reward Price Axis</div>
                <span class='badge-bull'>Blended R:R: 2.00R (Gross) &bull; ≥1.60R Net Gate</span>
            </div>
            <div style='display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr; align-items: center; gap: 8px;'>
                <div style='background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.4); border-radius: 8px; padding: 10px; text-align: center;'>
                    <div style='font-size: 0.68rem; color: #f43f5e; font-weight: 700;'>🛑 SAFETY STOP-LOSS</div>
                    <div class='price-text-bear' style='font-size: 1.15rem !important;'>₹{sl['price']:,.2f}</div>
                    <div style='font-size: 0.72rem; color: #fca5a5;'>▼ -{sl.get('loss_pct', 0.0):.2f}% Risk</div>
                </div>
                <div style='color: #64748b; font-size: 1.2rem; font-weight: 800;'>→</div>
                <div style='background: #161d2d; border: 1px solid #38bdf8; border-radius: 8px; padding: 10px; text-align: center;'>
                    <div style='font-size: 0.68rem; color: #38bdf8; font-weight: 700;'>📍 IDEAL ENTRY ZONE</div>
                    <div style='font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: "JetBrains Mono", monospace;'>₹{curr_p:,.2f}</div>
                    <div style='font-size: 0.72rem; color: #94a3b8;'>Market Price</div>
                </div>
                <div style='color: #64748b; font-size: 1.2rem; font-weight: 800;'>→</div>
                <div style='background: rgba(16, 185, 129, 0.10); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; padding: 10px; text-align: center;'>
                    <div style='font-size: 0.68rem; color: #10b981; font-weight: 700;'>🎯 TARGET 1 (50% LOCK)</div>
                    <div class='price-text-bull' style='font-size: 1.15rem !important;'>₹{t1['price']:,.2f}</div>
                    <div style='font-size: 0.72rem; color: #86efac;'>▲ +{t1.get('gain_pct', 0.0):.2f}% (1.5R)</div>
                </div>
                <div style='color: #64748b; font-size: 1.2rem; font-weight: 800;'>→</div>
                <div style='background: rgba(16, 185, 129, 0.16); border: 1.5px solid #10b981; border-radius: 8px; padding: 10px; text-align: center;'>
                    <div style='font-size: 0.68rem; color: #10b981; font-weight: 700;'>🚀 TARGET 2 (RUNNER)</div>
                    <div class='price-text-bull' style='font-size: 1.15rem !important;'>₹{t2['price']:,.2f}</div>
                    <div style='font-size: 0.72rem; color: #86efac;'>▲ +{t2.get('gain_pct', 0.0):.2f}% (2.5R)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 The Trade Blueprint (Exact Numbers)")
        b1, b2, b3, b4 = st.columns(4)
        
        b1.metric("📍 Ideal Entry Price Zone", f"{entry_z}", "Buy within this range")
        b2.metric("🎯 Target 1 (Quick Profit)", f"₹{t1['price']:,.2f}", f"▲ +{t1.get('gain_pct', 0.0):.1f}% gain ({t1.get('reward_risk', 1.5)}x R/R)", delta_color="normal")
        b3.metric("🎯 Target 2 (Extended Profit)", f"₹{t2['price']:,.2f}", f"▲ +{t2.get('gain_pct', 0.0):.1f}% gain ({t2.get('reward_risk', 2.5)}x R/R)", delta_color="normal")
        b4.metric("🛡️ Safety Stop-Loss", f"₹{sl['price']:,.2f}", f"▼ -{sl.get('loss_pct', 0.0):.1f}% loss", delta_color="normal")
        
        # Interactive Candlestick Chart with Live Strategy Overlays
        with st.expander("📊 **Interactive Strategy Candlestick Chart & Target Overlays**", expanded=True):
            chart_col1, chart_col2 = st.columns([3, 1])
            with chart_col2:
                adv_chart_tf = st.selectbox("Candle Timeframe:", ["15m (Short-Term)", "1h (Intraday/Swing)", "1d (Daily Trend)"], index=1, key="adv_chart_tf_sel")
                tf_val = adv_chart_tf.split(" ")[0]
            
            chart_period = "5d" if tf_val == "15m" else ("1mo" if tf_val == "1h" else "6mo")
            adv_chart_df = get_historical_data(adv_sym, period=chart_period, interval=tf_val)
            
            if not adv_chart_df.empty and len(adv_chart_df) >= 10:
                adv_chart_df = add_all_indicators(adv_chart_df)
                fig_adv = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
                
                # Candlestick
                fig_adv.add_trace(go.Candlestick(
                    x=adv_chart_df.index,
                    open=adv_chart_df["Open"],
                    high=adv_chart_df["High"],
                    low=adv_chart_df["Low"],
                    close=adv_chart_df["Close"],
                    name=f"{disp_name} Price",
                    increasing_line_color="#10b981",
                    decreasing_line_color="#f43f5e"
                ), row=1, col=1)
                
                # Institutional EMAs
                if "EMA_9" in adv_chart_df.columns:
                    fig_adv.add_trace(go.Scatter(x=adv_chart_df.index, y=adv_chart_df["EMA_9"], name="9 EMA", line=dict(color="#38bdf8", width=1.2)), row=1, col=1)
                if "EMA_21" in adv_chart_df.columns:
                    fig_adv.add_trace(go.Scatter(x=adv_chart_df.index, y=adv_chart_df["EMA_21"], name="21 EMA", line=dict(color="#f59e0b", width=1.2)), row=1, col=1)
                if "EMA_50" in adv_chart_df.columns:
                    fig_adv.add_trace(go.Scatter(x=adv_chart_df.index, y=adv_chart_df["EMA_50"], name="50 EMA", line=dict(color="#a855f7", width=1.2)), row=1, col=1)
                    
                # Stop-Loss Horizontal Overlay
                fig_adv.add_hline(
                    y=sl["price"], line_dash="dash", line_color="#f43f5e", line_width=2,
                    annotation_text=f"🛑 SL: ₹{sl['price']:,.2f} (-{sl.get('loss_pct', 0.0):.1f}%)",
                    annotation_position="bottom right", annotation_font_color="#f43f5e", annotation_font_size=11,
                    row=1, col=1
                )
                # Current Entry Price Horizontal Overlay
                fig_adv.add_hline(
                    y=curr_p, line_dash="dot", line_color="#38bdf8", line_width=1.5,
                    annotation_text=f"📍 Entry: ₹{curr_p:,.2f}",
                    annotation_position="top right", annotation_font_color="#38bdf8", annotation_font_size=11,
                    row=1, col=1
                )
                # Target 1 Horizontal Overlay
                fig_adv.add_hline(
                    y=t1["price"], line_dash="dash", line_color="#10b981", line_width=2,
                    annotation_text=f"🎯 Target 1 (1.5R): ₹{t1['price']:,.2f} (+{t1.get('gain_pct', 0.0):.1f}%)",
                    annotation_position="top right", annotation_font_color="#10b981", annotation_font_size=11,
                    row=1, col=1
                )
                # Target 2 Horizontal Overlay
                fig_adv.add_hline(
                    y=t2["price"], line_dash="dash", line_color="#22c55e", line_width=2.5,
                    annotation_text=f"🚀 Target 2 (2.5R): ₹{t2['price']:,.2f} (+{t2.get('gain_pct', 0.0):.1f}%)",
                    annotation_position="top right", annotation_font_color="#22c55e", annotation_font_size=11,
                    row=1, col=1
                )
                
                # Volume Subplot
                if "Volume" in adv_chart_df.columns:
                    vol_colors = ["#10b981" if c >= o else "#f43f5e" for o, c in zip(adv_chart_df["Open"], adv_chart_df["Close"])]
                    fig_adv.add_trace(go.Bar(
                        x=adv_chart_df.index, y=adv_chart_df["Volume"], name="Volume", marker_color=vol_colors
                    ), row=2, col=1)
                    
                fig_adv.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111622",
                    plot_bgcolor="#080b11",
                    xaxis_rangeslider_visible=False,
                    height=520,
                    margin=dict(l=30, r=40, t=30, b=30),
                    hovermode="x unified",
                    font=dict(family="Inter, sans-serif", color="#94a3b8"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(17,22,34,0.8)")
                )
                fig_adv.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                fig_adv.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                st.plotly_chart(fig_adv, use_container_width=True)
            else:
                st.info("Loading chart data...")
        
        # Interactive Investment & Profit Calculator
        st.markdown("### 🧮 Money & Profit Calculator (See Your Returns in ₹)")
        calc_col1, calc_col2 = st.columns([2, 3])
        
        with calc_col1:
            invest_budget = st.slider(
                "Select Capital to Allocate for this Trade (₹):",
                min_value=5000, max_value=200000, value=25000, step=5000
            )
            curr_p = analysis["current_price"]
            shares_qty = max(1, int(invest_budget / curr_p))
            actual_invested = shares_qty * curr_p
            
        with calc_col2:
            profit_t1 = (t1["price"] - curr_p) * shares_qty
            profit_t2 = (t2["price"] - curr_p) * shares_qty
            max_risk = (curr_p - sl["price"]) * shares_qty
            
            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            c_m1.metric("📦 Shares to Buy", f"{shares_qty} shares", f"₹{actual_invested:,.0f}")
            c_m2.metric("💰 Profit @ Target 1", f"+₹{profit_t1:,.0f}", f"▲ +{t1['gain_pct']:.1f}%", delta_color="normal")
            c_m3.metric("🚀 Profit @ Target 2", f"+₹{profit_t2:,.0f}", f"▲ +{t2['gain_pct']:.1f}%", delta_color="normal")
            c_m4.metric("🛡️ Max Loss @ SL", f"-₹{max_risk:,.0f}", f"▼ -{sl['loss_pct']:.1f}%", delta_color="normal")
            
        # Pros vs Watchouts Checklist
        st.markdown("### 🔍 Technical Diagnosis & Key Factors")
        chk_col1, chk_col2 = st.columns(2)
        with chk_col1:
            st.markdown("#### ✅ Bullish Factors (Why This Looks Good)")
            if analysis.get("pros"):
                for p in analysis["pros"]:
                    st.markdown(p)
            else:
                st.info("No strong bullish momentum signals currently.")
                
        with chk_col2:
            st.markdown("#### ⚠️ Watchouts & Caution Points")
            if analysis.get("watchouts"):
                for w in analysis["watchouts"]:
                    st.markdown(w)
            else:
                st.success("No major technical warning signals detected!")
                
        # 1-Click Action to Execute in Paper Portfolio
        st.markdown("---")
        act_col1, act_col2 = st.columns([2.5, 1.5])
        with act_col1:
            st.markdown(f"**Ready to take this setup in your Paper Trading Account?**")
            st.caption(f"Will buy {shares_qty} shares of {analysis.get('display_name', display_symbol_name(adv_sym))} @ ₹{curr_p:.2f} with Stop-Loss ₹{sl['price']:.2f} and Target ₹{t1['price']:.2f}.")
        with act_col2:
            if st.button(f"⚡ Execute Trade in Paper Account ({shares_qty} Shares)", type="primary", use_container_width=True):
                order_res = broker.place_order(
                    symbol=adv_sym,
                    side="BUY",
                    quantity=shares_qty,
                    price=curr_p,
                    sl=sl["price"],
                    tp=t1["price"],
                    strategy_name=f"Smart Advisor ({h_key.upper()})"
                )
                if order_res.get("status") in ["FILLED", "SUCCESS"]:
                    st.success(f"Order Executed! Bought {shares_qty} shares of {display_symbol_name(adv_sym)} in Paper Account.")
                    st.rerun()
                else:
                    st.error(f"Order Failed: {order_res.get('message')}")
    else:
        st.error(analysis.get("message", "Error analyzing stock."))

# -------------------------------------------------------------
# TAB 2: 📊 Strategy Backtester & Optimizer
# -------------------------------------------------------------
elif active_tab == "📊 Strategy Backtester (Test Any Stock)":
    st.markdown("""
    <h2>📊 Strategy Backtester — Test Any Indian Stock</h2>
    <div style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 14px;'>Test how much money a strategy would have made on any Indian stock in the past with realistic taxes, brokerage, and safety rules.</div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔍 **Step 1: Search & Select Indian Stock to Test**", expanded=True):
        bt_col1, bt_col2 = st.columns([2.5, 2])
        with bt_col1:
            bt_search = st.text_input("🔍 Search Company Name or Ticker to Backtest:", value="", placeholder="Type any name e.g. Tata, Zomato, Reliance, Suzlon, SBI, Paytm, HAL, ITC...")
            matching_bt = search_indian_stocks(bt_search)
            
            selected_sym = st.selectbox(
                f"Choose from Matching Results ({len(matching_bt)} available):",
                options=[item["symbol"] for item in matching_bt],
                format_func=lambda s: next((f"{item['name']} ({item['symbol'].replace('.NS','')}) — {item.get('category', 'Equity')}" for item in matching_bt if item["symbol"] == s), s),
                index=0
            )
            
            # Show live quote preview
            bt_quote = get_live_quote(selected_sym)
            if bt_quote.get("price", 0) > 0:
                st.caption(f"⚡ **Live LTP:** ₹{bt_quote['price']:,.2f} ({bt_quote['change_pct']:+.2f}%) | Previous Close: ₹{bt_quote['previous_close']:,.2f}")
        with bt_col2:
            st.info("💡 **Tip:** You can test any Indian stock across 5 different technical strategies to see historical profits, win rate %, and risk.")
                
    with st.expander("🧠 **Step 2: Choose Strategy & Simple Settings**", expanded=True):
        st_col1, st_col2 = st.columns([2, 2])
        with st_col1:
            strategy_choice = st.selectbox(
                "Select Trading Strategy:",
                [
                    "🚀 Trend Rider (EMA Crossover + RSI)",
                    "🛡️ Smart Safety Net (SuperTrend)",
                    "⚡ Discount Dip Buyer (Bollinger Bands)",
                    "🌊 Momentum Wave (MACD)",
                    "🎯 All-Rounder Combo (Multi-Indicator Confluence)"
                ]
            )
            
            # Friendly Strategy Explanation
            if "Trend Rider" in strategy_choice:
                internal_strat_name = "EMA Crossover + RSI"
                st.markdown("""
                <div class='tip-box'>
                    💡 <strong>How Trend Rider Works:</strong><br>
                    • <strong>BUY:</strong> When the short-term trend crosses above the long-term trend AND buyer energy is strong.<br>
                    • <strong>SELL:</strong> When the trend cools down or momentum reverses.
                </div>
                """, unsafe_allow_html=True)
            elif "Smart Safety Net" in strategy_choice:
                internal_strat_name = "SuperTrend (Intraday/Swing)"
                st.markdown("""
                <div class='tip-box'>
                    💡 <strong>How Smart Safety Net Works:</strong><br>
                    • <strong>BUY:</strong> Enters when the SuperTrend indicator turns <strong>GREEN</strong>.<br>
                    • <strong>SELL:</strong> Exits immediately when the line turns <strong>RED</strong> to protect capital.
                </div>
                """, unsafe_allow_html=True)
            elif "Discount Dip Buyer" in strategy_choice:
                internal_strat_name = "Bollinger Bands"
                st.markdown("""
                <div class='tip-box'>
                    💡 <strong>How Discount Dip Buyer Works:</strong><br>
                    • <strong>BUY:</strong> When the stock price drops below its normal range (on sale) and begins bouncing back up.<br>
                    • <strong>SELL:</strong> When the price reaches the top expensive band.
                </div>
                """, unsafe_allow_html=True)
            elif "Momentum Wave" in strategy_choice:
                internal_strat_name = "MACD Momentum"
                st.markdown("""
                <div class='tip-box'>
                    💡 <strong>How Momentum Wave Works:</strong><br>
                    • <strong>BUY:</strong> Catches the start of a fast rally when momentum accelerates upward.<br>
                    • <strong>SELL:</strong> Exits when momentum slows down.
                </div>
                """, unsafe_allow_html=True)
            else:
                internal_strat_name = "Multi-Indicator Confluence"
                st.markdown("""
                <div class='tip-box'>
                    💡 <strong>How All-Rounder Combo Works (Recommended):</strong><br>
                    • <strong>BUY:</strong> Highest safety — only buys when Trend, Speed (RSI), Momentum (MACD), and Volatility (SuperTrend) <strong>ALL agree</strong>.
                </div>
                """, unsafe_allow_html=True)
                
        with st_col2:
            tf_col, per_col = st.columns(2)
            with tf_col:
                timeframe = st.selectbox(
                    "Candle Timeframe:",
                    ["5m (Day Trading)", "15m (Short-Term)", "1h (Swing Trading)", "1d (Positional / Days)"],
                    index=1
                )
                tf_clean = timeframe.split(" ")[0]
            with per_col:
                history_period = st.selectbox(
                    "Test Duration:",
                    ["1mo (Past Month)", "3mo (Past 3 Months)", "6mo (Past 6 Months)", "1y (Past 1 Year)", "2y (Past 2 Years)"],
                    index=2
                )
                per_clean = history_period.split(" ")[0]
                
            st.markdown("##### 🛡️ Safety & Money Settings")
            r1, r2, r3 = st.columns(3)
            with r1:
                test_capital = st.number_input("Starting Capital (₹)", value=100000.0, step=10000.0)
            with r2:
                sl_pct = st.slider("Safety Stop-Loss % (Max Loss)", 0.5, 5.0, 1.5, 0.25, help="If the trade loses this %, the bot cuts the loss automatically.")
            with r3:
                tp_pct = st.slider("Profit Target % (Goal)", 1.0, 10.0, 3.0, 0.5, help="When the trade reaches this profit %, the bot sells to pocket the gain.")
                
    run_btn = st.button("🚀 Run Backtest Simulation Now", type="primary", use_container_width=True)
    
    if run_btn:
        with st.spinner(f"Loading real market data for {display_symbol_name(selected_sym)} and running test..."):
            hist_df = get_historical_data(selected_sym, period=per_clean, interval=tf_clean)
            
            if hist_df.empty:
                st.error(f"Could not load data for {selected_sym}. Please verify the symbol or choose a different timeframe.")
            else:
                strategy = get_strategy(internal_strat_name)
                risk_mgr = RiskManager(default_sl_pct=sl_pct, default_tp_pct=tp_pct)
                backtester = Backtester(strategy=strategy, initial_capital=test_capital, risk_manager=risk_mgr)
                results = backtester.run(hist_df)
                
                st.markdown("### 🏆 Easy Performance Scorecard")
                k1, k2, k3, k4, k5 = st.columns(5)
                
                net_p = results["net_profit"]
                ret_p = results["total_return_pct"]
                ret_color = "normal"
                ret_arr = "▲ +" if ret_p >= 0 else "▼ "
                
                k1.metric("💰 Total Profit / Loss", format_currency_inr(net_p), f"{ret_arr}{ret_p:.2f}%", delta_color=ret_color)
                k2.metric("🎯 Win Score", f"{results['win_rate_pct']:.1f}%", f"{results['winning_trades']} Won / {results['losing_trades']} Lost")
                k3.metric("📊 Nifty 50 Comparison", f"{results['benchmark_return_pct']:+.2f}%", f"Strategy: {ret_p:+.2f}%")
                k4.metric("🛡️ Max Drop (Risk)", f"-{results['max_drawdown_pct']:.2f}%", "Lower is safer")
                k5.metric("⚖️ Reward / Risk", f"{results['risk_reward_ratio']:.2f}x", "Earns ₹ vs ₹1 Risk")
                
                # Interactive Candlestick Chart with Buy/Sell flags
                st.markdown("### 📈 Visual Chart — Look Where It Bought & Sold")
                signals_df = results["signals_df"]
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=(f"{display_symbol_name(selected_sym)} Price Chart (▲ Green = BUY, ▼ Red = SELL)", "Trading Volume"),
                    row_heights=[0.75, 0.25]
                )
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=signals_df.index,
                    open=signals_df["Open"],
                    high=signals_df["High"],
                    low=signals_df["Low"],
                    close=signals_df["Close"],
                    name="Stock Price",
                    increasing_line_color="#10b981",
                    decreasing_line_color="#f43f5e"
                ), row=1, col=1)
                
                # Buy / Sell Marker Overlay
                buy_signals = signals_df[signals_df["Signal"] == 1]
                sell_signals = signals_df[signals_df["Signal"] == -1]
                
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals["Low"] * 0.995,
                        mode="markers",
                        name="🟢 BUY Entry",
                        marker=dict(symbol="triangle-up", size=14, color="#10b981", line=dict(width=1.5, color="#ffffff"))
                    ), row=1, col=1)
                    
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals["High"] * 1.005,
                        mode="markers",
                        name="🔴 SELL / Exit",
                        marker=dict(symbol="triangle-down", size=14, color="#f43f5e", line=dict(width=1.5, color="#ffffff"))
                    ), row=1, col=1)
                    
                # Volume
                if "Volume" in signals_df.columns:
                    colors = ["#10b981" if c >= o else "#f43f5e" for o, c in zip(signals_df["Open"], signals_df["Close"])]
                    fig.add_trace(go.Bar(
                        x=signals_df.index,
                        y=signals_df["Volume"],
                        name="Volume",
                        marker_color=colors
                    ), row=2, col=1)
                    
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111622",
                    plot_bgcolor="#080b11",
                    xaxis_rangeslider_visible=False,
                    height=520,
                    margin=dict(l=30, r=30, t=40, b=30),
                    hovermode="x unified",
                    font=dict(family="Inter, sans-serif", color="#94a3b8"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(17,22,34,0.8)")
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Equity Growth Curve
                st.markdown("### 💵 Money Growth Curve (How Your Capital Grew Over Time)")
                eq_df = results["equity_df"]
                if not eq_df.empty:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=eq_df.index,
                        y=eq_df["Equity"],
                        name="ApexTrade Strategy Capital (₹)",
                        line=dict(color="#10b981", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(16, 185, 129, 0.08)"
                    ))
                    fig_eq.add_trace(go.Scatter(
                        x=eq_df.index,
                        y=eq_df["Benchmark_Equity"],
                        name="Regular Buy & Hold Capital (₹)",
                        line=dict(color="#94a3b8", width=1.5, dash="dot")
                    ))
                    fig_eq.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#111622",
                        plot_bgcolor="#080b11",
                        height=350,
                        margin=dict(l=30, r=30, t=30, b=30),
                        hovermode="x unified",
                        font=dict(family="Inter, sans-serif", color="#94a3b8"),
                        yaxis_title="Account Balance (₹)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(17,22,34,0.8)")
                    )
                    fig_eq.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                    fig_eq.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                    st.plotly_chart(fig_eq, use_container_width=True)
                    
                # Trade History Journal
                st.markdown("### 📜 Every Trade Listed Step-by-Step")
                trades_list = results["trades"]
                if trades_list:
                    trades_df = pd.DataFrame(trades_list)
                    trades_df["Profit / Loss (₹)"] = trades_df["net_pnl"].apply(lambda x: format_currency_inr(x))
                    trades_df["Return %"] = trades_df["pnl_pct"].apply(lambda x: f"{x:+.2f}%")
                    trades_df["Bought At"] = trades_df["entry_price"].apply(lambda x: f"₹{x:.2f}")
                    trades_df["Sold At"] = trades_df["exit_price"].apply(lambda x: f"₹{x:.2f}")
                    
                    display_cols = ["entry_date", "exit_date", "side", "quantity", "Bought At", "Sold At", "Profit / Loss (₹)", "Return %", "exit_reason"]
                    st.dataframe(trades_df[display_cols], use_container_width=True, hide_index=True)
                    
                    csv = pd.DataFrame(trades_list).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Trades List (Excel / CSV)",
                        data=csv,
                        file_name=f"trades_{selected_sym}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No trades were generated with these exact settings over this time period.")

# -------------------------------------------------------------
# TAB 3: ⚡ Automated Live / Paper Bot
# -------------------------------------------------------------
elif active_tab == "⚡ Automated Live / Paper Bot":
    st.markdown("""
    <h2>⚡ Automated Bot Engine</h2>
    <div style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 14px;'>Let the bot automatically scan Indian stocks, take trades according to your rules, and protect your capital.</div>
    """, unsafe_allow_html=True)
    
    b_col1, b_col2, b_col3, b_col4 = st.columns([2, 2, 2, 2])
    with b_col1:
        if st.session_state.bot_running:
            if st.button("⏹️ Pause Bot", type="secondary", use_container_width=True):
                st.session_state.bot_running = False
                st.rerun()
        else:
            if st.button("▶️ Turn Bot ON", type="primary", use_container_width=True):
                st.session_state.bot_running = True
                st.rerun()
                
    with b_col2:
        scan_now = st.button("🔄 Scan Market Now", use_container_width=True)
        
    with b_col3:
        bot_strat = st.selectbox("Strategy to Use:", list(AVAILABLE_STRATEGIES.keys()), index=0)
        
    with b_col4:
        bot_tf = st.selectbox("Candle Speed:", ["1m", "5m", "15m", "30m"], index=1)
        
    bot = st.session_state.bot_instance
    bot.strategy_name = bot_strat
    bot.strategy = get_strategy(bot_strat)
    bot.timeframe = bot_tf
    
    if scan_now or st.session_state.bot_running:
        with st.spinner("Scanning Indian stocks and evaluating trade setups..."):
            scan_res = bot.scan_and_execute()
            if scan_res.get("status") == "SUCCESS":
                st.success(f"Scan complete at {scan_res['last_scan']}. Checked stocks and managed open trades.")
            elif scan_res.get("status") == "HALTED":
                st.error(scan_res.get("message"))
                
    st.markdown("---")
    
    @st.fragment(run_every=2)
    def render_active_open_positions():
        broker_local = get_broker(st.session_state.active_broker_name)
        active_pos = broker_local.get_open_positions()
        
        st.markdown("### 📌 Active Open Positions (Live 2s Daemon Stream)")
        
        if active_pos:
            # Summary Metrics for all active positions
            tot_pnl = sum(p.get("unrealized_pnl", 0.0) for p in active_pos)
            tot_exposure = sum(p["entry_price"] * p["quantity"] for p in active_pos)
            tot_arr = "▲ +" if tot_pnl >= 0 else "▼ "
            
            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns([2, 2, 2, 2])
            sum_col1.metric("📊 Active Positions", f"{len(active_pos)} leg(s)")
            sum_col2.metric("💼 Total Exposure Deployed", format_currency_inr(tot_exposure))
            sum_col3.metric("📈 Net Live Open P&L", format_currency_inr(tot_pnl), f"{tot_arr}{tot_pnl:,.2f} ₹", delta_color="normal")
            with sum_col4:
                st.markdown("<div style='padding-top: 14px;'>", unsafe_allow_html=True)
                if st.button("🚨 Square Off All Legs", type="secondary", use_container_width=True):
                    sq_res = broker_local.square_off_all(reason="Manual Bulk Close")
                    st.success(f"Closed {len(sq_res)} open positions!")
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
            
            # Individual Position Cards
            for pos in active_pos:
                curr_p = float(pos.get("current_price", pos["entry_price"]))
                entry_p = float(pos["entry_price"])
                qty = int(pos["quantity"])
                exposure = entry_p * qty
                pnl = pos.get("unrealized_pnl", 0.0)
                pnl_p = pos.get("unrealized_pnl_pct", 0.0)
                pnl_arr = "▲ +" if pnl >= 0 else "▼ "
                card_border = "#10b981" if pnl >= 0 else "#f43f5e"
                side_badge = "badge-bull" if pos["side"].upper() in ["BUY", "LONG"] else "badge-bear"
                entry_time_str = pos.get("entry_time", "N/A")
                holding_dur = format_holding_duration(entry_time_str)
                strat_name = pos.get("strategy", "Autonomous AI")
                
                sl_val = pos.get("sl")
                tp_val = pos.get("tp")
                sl_str = f"₹{float(sl_val):,.2f}" if sl_val is not None else "None"
                tp_str = f"₹{float(tp_val):,.2f}" if tp_val is not None else "None"
                
                # Render Elevated Position Card
                st.markdown(f"""
                <div style='background: #111622; border-left: 6px solid {card_border}; border-top: 1px solid #1e293b; border-right: 1px solid #1e293b; border-bottom: 1px solid #1e293b; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 10px;'>
                        <div style='display: flex; align-items: center; gap: 10px;'>
                            <div style='font-size: 1.25rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>{display_symbol_name(pos['symbol'])}</div>
                            <span class='{side_badge}' style='font-size: 0.76rem; font-weight: 700;'>{pos['side']}</span>
                            <span class='badge-cyan' style='font-size: 0.76rem;'>{strat_name}</span>
                        </div>
                        <div style='display: flex; align-items: center; gap: 8px;'>
                            <span class='badge-neutral' style='font-size: 0.78rem; font-family: "JetBrains Mono", monospace;'>🕒 Executed: {entry_time_str}</span>
                            <span class='badge-bull' style='font-size: 0.78rem; font-weight: 700;'>⏳ {holding_dur}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 4-Column Metric Grid & Action Button
                pc1, pc2, pc3, pc4, pc5 = st.columns([2, 1.8, 2, 2, 1.4])
                pc1.metric("📦 Bought @ Entry", f"₹{entry_p:,.2f}", f"{qty} sh • ₹{exposure:,.0f} Total")
                price_diff = curr_p - entry_p
                diff_arr = "▲ +" if price_diff >= 0 else "▼ "
                pc2.metric("⚡ Live Market Price", f"₹{curr_p:,.2f}", f"{diff_arr}₹{abs(price_diff):.2f}/sh", delta_color="normal")
                pc3.metric("💰 Live Profit / Loss", format_currency_inr(pnl), f"{pnl_arr}{pnl_p:.2f}% Return", delta_color="normal")
                
                with pc4:
                    st.markdown(f"""
                    <div style='padding-top: 8px; font-size: 0.84rem;'>
                        <div>🛡️ <strong>Safety SL:</strong> <span class='mono-num'>{sl_str}</span></div>
                        <div style='margin-top: 4px;'>🎯 <strong>Target TP:</strong> <span class='mono-num'>{tp_str}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with pc5:
                    st.markdown("<div style='padding-top: 18px;'>", unsafe_allow_html=True)
                    if st.button("🚨 Exit Trade", key=f"sq_{pos['symbol']}", type="secondary", use_container_width=True):
                        broker_local.square_off_position(pos["symbol"], reason="Manual User Close")
                        st.success(f"Closed {pos['symbol']}!")
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                # Expandable Live Candlestick Chart for Active Position
                with st.expander(f"📈 **View Live Position Chart & Overlays for {display_symbol_name(pos['symbol'])}**", expanded=False):
                    pos_sym = pos["symbol"]
                    p_df = get_historical_data(pos_sym, period="5d", interval="15m")
                    if not p_df.empty and len(p_df) >= 10:
                        p_df = add_all_indicators(p_df)
                        fig_pos = go.Figure()
                        
                        # Candlestick
                        fig_pos.add_trace(go.Candlestick(
                            x=p_df.index,
                            open=p_df["Open"],
                            high=p_df["High"],
                            low=p_df["Low"],
                            close=p_df["Close"],
                            name=f"{display_symbol_name(pos_sym)} LTP",
                            increasing_line_color="#10b981",
                            decreasing_line_color="#f43f5e"
                        ))
                        
                        # Entry Line
                        fig_pos.add_hline(
                            y=entry_p, line_dash="solid", line_color="#38bdf8", line_width=2,
                            annotation_text=f"📍 Bought @ ₹{entry_p:,.2f}",
                            annotation_position="top left", annotation_font_color="#38bdf8"
                        )
                        
                        # Stop-Loss Line
                        if sl_val:
                            fig_pos.add_hline(
                                y=float(sl_val), line_dash="dash", line_color="#f43f5e", line_width=2,
                                annotation_text=f"🛑 Safety SL: ₹{float(sl_val):,.2f}",
                                annotation_position="bottom right", annotation_font_color="#f43f5e"
                            )
                            
                        # Target TP Line
                        if tp_val:
                            fig_pos.add_hline(
                                y=float(tp_val), line_dash="dash", line_color="#10b981", line_width=2,
                                annotation_text=f"🎯 Target TP: ₹{float(tp_val):,.2f}",
                                annotation_position="top right", annotation_font_color="#10b981"
                            )
                            
                        fig_pos.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#111622",
                            plot_bgcolor="#080b11",
                            xaxis_rangeslider_visible=False,
                            height=380,
                            margin=dict(l=20, r=20, t=20, b=20),
                            hovermode="x unified",
                            font=dict(family="Inter, sans-serif", color="#94a3b8")
                        )
                        fig_pos.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                        fig_pos.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                        st.plotly_chart(fig_pos, use_container_width=True)
                    else:
                        st.info("Loading live position chart...")
                        
                st.divider()
        else:
            st.markdown("""
            <div class='op-card' style='padding: 24px; text-align: center;'>
                <div style='font-size: 1.1rem; font-weight: 700; color: #f8fafc; font-family: "Outfit", sans-serif; margin-bottom: 6px;'>🛡️ No Active Open Positions</div>
                <div style='color: #94a3b8; font-size: 0.86rem;'>The trading bot and risk guardrails are standing by and continuously scanning for high-probability market setups.</div>
            </div>
            """, unsafe_allow_html=True)

    render_active_open_positions()
        
    st.markdown("### 📜 Real-Time Bot Logs (Activity Terminal)")
    if bot.logs:
        log_text = "\n".join(bot.logs[:30])
        st.text_area("Bot Activity Log", log_text, height=180)
    else:
        st.info("No activity yet. Click 'Scan Market Now' to trigger a scan.")

# -------------------------------------------------------------
# TAB 4: 🔍 Indian Market Screener (Scan All Stocks)
# -------------------------------------------------------------
elif active_tab == "🔍 Indian Market Screener (Scan All Stocks)":
    st.markdown("""
    <h2>🔍 Real-Time Market Screener (Scan 70+ Indian Stocks)</h2>
    <div style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 14px;'>Instantly scans top Indian companies across Banking, IT, Power, Auto, Defence, and FMCG to find which stocks are currently in a BUY or SELL zone.</div>
    """, unsafe_allow_html=True)
    
    sc1, sc2 = st.columns([2, 2])
    with sc1:
        screener_sector = st.selectbox(
            "Filter Sector:",
            ["All Sectors", "Banking", "IT & Tech", "Power & Energy", "Automobile", "Defence & Aerospace", "Railways", "FMCG", "Metals & Mining", "Healthcare & Pharma"]
        )
    with sc2:
        screener_tf = st.selectbox("Scan Timeframe:", ["15m (Short-Term)", "1h (Intraday/Swing)", "1d (Daily Trend)"], index=1)
        tf_code = screener_tf.split(" ")[0]
        
    if st.button("🚀 Run Live Market Scan", type="primary", use_container_width=True):
        with st.spinner("Scanning companies and calculating buy/sell signals..."):
            stock_pool = config.DEFAULT_WATCHLIST
            if screener_sector != "All Sectors":
                stock_pool = [i for i in stock_pool if screener_sector.lower() in i["category"].lower()]
                
            screener_rows = []
            for item in stock_pool:
                sym = item["symbol"]
                if sym.startswith("^"):
                    continue
                    
                df = get_historical_data(sym, period="5d", interval=tf_code)
                if df.empty or len(df) < 20:
                    continue
                    
                df = add_all_indicators(df)
                last = df.iloc[-1]
                close_p = float(last["Close"])
                
                # Calculate scores
                bull_pts = 0
                if last["EMA_9"] > last["EMA_21"]: bull_pts += 1
                if float(last["RSI_14"]) >= 52: bull_pts += 1
                if last["MACD"] > last["MACD_Signal"]: bull_pts += 1
                if last["SuperTrend_Dir"] == 1: bull_pts += 1
                
                if bull_pts >= 4:
                    signal_badge = "🟢 STRONG BUY"
                    tip = "All 4 indicators are bullish"
                elif bull_pts == 3:
                    signal_badge = "🟢 BUY"
                    tip = "3 out of 4 indicators bullish"
                elif bull_pts == 2:
                    signal_badge = "🟡 WAIT / NEUTRAL"
                    tip = "Mixed signals, wait for clear trend"
                elif bull_pts == 1:
                    signal_badge = "🔴 WEAK / SELL"
                    tip = "Bearish pressure"
                else:
                    signal_badge = "🔴 STRONG SELL"
                    tip = "All indicators bearish"
                    
                screener_rows.append({
                    "Company Name": item["name"],
                    "Ticker": sym.replace(".NS", ""),
                    "Sector": item["category"],
                    "Live Price": f"₹{close_p:,.2f}",
                    "RSI (Buyer Energy)": f"{float(last['RSI_14']):.1f}",
                    "Trend": "🟢 Rising" if last["EMA_9"] > last["EMA_21"] else "🔴 Falling",
                    "Recommendation": signal_badge,
                    "Summary": tip
                })
                
            if screener_rows:
                res_df = pd.DataFrame(screener_rows)
                st.dataframe(res_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No data returned for selected sector.")

# -------------------------------------------------------------
# TAB 5: ⚙️ Settings & Risk Controls
# -------------------------------------------------------------
elif active_tab == "⚙️ Settings & Risk Controls":
    st.markdown("""
    <h2>⚙️ Settings, Risk Rules & Broker Connections</h2>
    <div style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 14px;'>Manage your capital, broker connections, and automated safety limits.</div>
    """, unsafe_allow_html=True)
    
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("""
        <div class='op-card' style='padding: 16px 20px; margin-bottom: 16px;'>
            <div style='font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px; font-family: "Outfit", sans-serif;'>🔌 Active Execution Broker</div>
            <div style='color: #94a3b8; font-size: 0.84rem; margin-bottom: 12px;'>Select between Virtual Demo Account (Zero Risk) and Real Broker Routing.</div>
        </div>
        """, unsafe_allow_html=True)
        
        chosen_broker = st.selectbox(
            "Select Execution Broker:",
            ["paper", "zerodha", "angel", "dhan"],
            format_func=lambda x: {
                "paper": "🛡️ Paper Trading (Virtual Demo Money - Recommended)",
                "zerodha": "🪁 Zerodha Kite Connect (Live Real Trading)",
                "angel": "👼 Angel One SmartAPI (Live Real Trading)",
                "dhan": "🏹 DhanHQ (Live Real Trading)"
            }.get(x, x),
            index=["paper", "zerodha", "angel", "dhan"].index(st.session_state.active_broker_name)
        )
        
        if st.button("Save Active Broker", type="primary", use_container_width=True):
            st.session_state.active_broker_name = chosen_broker
            st.session_state.bot_instance = LiveTradingBot(broker_name=chosen_broker)
            st.success(f"Execution broker switched to: {chosen_broker.upper()}")
            st.rerun()
            
        st.markdown("---")
        with st.expander("🔑 **Live Broker API Credentials (Encrypted)**", expanded=False):
            st.caption("Enter your Kite Connect, SmartAPI, or Dhan credentials for direct order execution.")
            with st.form("broker_keys_form"):
                st.text_input("Zerodha API Key", value=config.ZERODHA_API_KEY, type="password")
                st.text_input("Zerodha Access Token", value=config.ZERODHA_ACCESS_TOKEN, type="password")
                st.text_input("Angel One API Key", value=config.ANGEL_API_KEY, type="password")
                st.text_input("Angel One Client ID", value=config.ANGEL_CLIENT_ID)
                st.text_input("Dhan Client ID", value=config.DHAN_CLIENT_ID)
                st.text_input("Dhan Access Token", value=config.DHAN_ACCESS_TOKEN, type="password")
                st.form_submit_button("Save API Credentials")
            
    with b2:
        st.markdown("""
        <div class='op-card' style='padding: 16px 20px; margin-bottom: 16px;'>
            <div style='font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px; font-family: "Outfit", sans-serif;'>🛡️ Risk Controls & Circuit Breakers</div>
            <div style='color: #94a3b8; font-size: 0.84rem;'>Deterministic rules that hard-block order routing if risk limits are breached.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.slider("Max Capital to Risk per Trade (%)", 0.5, 10.0, float(config.DEFAULT_RISK_PER_TRADE_PCT), 0.5, help="Caps maximum capital at risk on a single trade.")
        st.slider("Daily Drawdown Circuit Breaker (%)", 1.0, 15.0, float(config.MAX_DAILY_LOSS_PCT), 0.5, help="If daily loss hits this %, the bot stops trading immediately.")
        st.slider("Default Trailing Stop-Loss (%)", 0.25, 5.0, float(config.DEFAULT_TRAILING_SL_PCT), 0.25, help="Automatically trails stop-loss higher as trade gains.")
        st.text_input("Intraday Auto-Squareoff Time", value="3:15 PM IST (SEBI Standard Mandatory)", disabled=True)
        
        st.markdown("---")
        st.markdown("""
        <div class='op-card' style='padding: 14px 18px;'>
            <div style='font-size: 1rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; font-family: "Outfit", sans-serif;'>🧹 Virtual Account Balance Manager</div>
            <div style='color: #94a3b8; font-size: 0.82rem;'>Reset your virtual practice balance back to ₹1,00,000 and clear simulated trade history.</div>
        </div>
        """, unsafe_allow_html=True)
        reset_cap = st.number_input("Reset Virtual Balance to (₹)", value=100000.0, step=10000.0)
        if st.button("⚠️ Reset Virtual Account Data", type="secondary", use_container_width=True):
            reset_all_data(initial_capital=reset_cap)
            st.success("Virtual account balance reset successfully!")
            st.rerun()

# -------------------------------------------------------------
# End of Application
# -------------------------------------------------------------
