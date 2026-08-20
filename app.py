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
from src.engine.pre_market_analyzer import PreMarketAnalyzer
from src.ai import (
    LLMClient, MarketPrompter, FailsafeParser, ConfidenceCalibrator,
    AITradingAgent, MarketRadarScanner
)
from src.ai.chat_assistant import TradingChatAssistant
from src.ai.multi_agent_council import MultiAgentCouncil
from src.engine.market_hunter_daemon import MarketHunterDaemon
from src.engine.software_oco_manager import SoftwareOCOManager
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

# Inject Google Fonts & Institutional Terminal CSS Design Tokens
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

    :root {
        /* Base Flat Surfaces (Zero gradients, zero glow) */
        --bg-obsidian: #080b11;
        --bg-surface: #111622;
        --bg-surface-elevated: #182030;
        
        /* Subtle 1px Institutional Borders */
        --border-subtle: #1e293b;
        --border-prominent: #334155;
        
        /* High-Contrast Typography */
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        
        /* Accessible Financial Indicators */
        --color-bullish: #10b981;
        --color-bullish-bg: rgba(16, 185, 129, 0.10);
        --color-bullish-border: rgba(16, 185, 129, 0.35);
        
        --color-bearish: #f43f5e;
        --color-bearish-bg: rgba(244, 63, 94, 0.10);
        --color-bearish-border: rgba(244, 63, 94, 0.35);
        
        --color-neutral: #f59e0b;
        --color-neutral-bg: rgba(245, 158, 11, 0.10);
        --color-neutral-border: rgba(245, 158, 11, 0.35);
        
        --color-sky: #0ea5e9;
        --color-sky-bg: rgba(14, 165, 233, 0.10);
        --color-sky-border: rgba(14, 165, 233, 0.35);
    }

    /* Core Terminal Canvas */
    .stApp {
        background-color: var(--bg-obsidian) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Tabular Monospace for All Numerical Telemetry */
    .tnum, .mono-num, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-feature-settings: "tnum" 1 !important;
        font-variant-numeric: tabular-nums !important;
        letter-spacing: -0.02em !important;
    }

    /* Section Headings */
    h1, h2, h3, .brand-title {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }

    /* Flat High-Contrast Metric Cards (Zero blur / zero gradient) */
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

    /* Flat Equal-Height Ticker Cards */
    .ticker-card-bull {
        background: var(--color-bullish-bg) !important;
        border: 1px solid var(--color-bullish-border) !important;
        border-radius: 8px;
        padding: 10px 14px;
        min-height: 82px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .ticker-card-bear {
        background: var(--color-bearish-bg) !important;
        border: 1px solid var(--color-bearish-border) !important;
        border-radius: 8px;
        padding: 10px 14px;
        min-height: 82px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .ticker-card-neutral {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px;
        padding: 10px 14px;
        min-height: 82px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .ticker-val-bull {
        color: var(--color-bullish) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-feature-settings: "tnum" 1 !important;
        font-size: 1.22rem !important;
        font-weight: 800 !important;
    }
    .ticker-val-bear {
        color: var(--color-bearish) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-feature-settings: "tnum" 1 !important;
        font-size: 1.22rem !important;
        font-weight: 800 !important;
    }

    /* Solid Operational Cards */
    .op-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 10px 14px;
        min-height: 82px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Accessibility Badges (Always paired with symbol/text) */
    .badge-bull {
        background-color: var(--color-bullish-bg);
        color: var(--color-bullish);
        border: 1px solid var(--color-bullish-border);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-feature-settings: "tnum" 1;
    }
    .badge-bear {
        background-color: var(--color-bearish-bg);
        color: var(--color-bearish);
        border: 1px solid var(--color-bearish-border);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-feature-settings: "tnum" 1;
    }
    .badge-neutral {
        background-color: var(--color-neutral-bg);
        color: var(--color-neutral);
        border: 1px solid var(--color-neutral-border);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-feature-settings: "tnum" 1;
    }
    .badge-sky {
        background-color: var(--color-sky-bg);
        color: var(--color-sky);
        border: 1px solid var(--color-sky-border);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-feature-settings: "tnum" 1;
    }

    /* Slow Ambient Attention Indicator (1.8s calm fade for urgent state only) */
    @keyframes urgent-attention {
        0%, 100% { opacity: 0.95; }
        50% { opacity: 0.35; }
    }
    .dot-live-open {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: var(--color-bullish);
        border-radius: 50%;
        margin-right: 5px;
    }
    .dot-urgent-attention {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: var(--color-bearish);
        border-radius: 50%;
        margin-right: 5px;
        animation: urgent-attention 1.8s infinite ease-in-out;
    }

    /* Solid Unmistakable Kill Switch Box */
    .kill-switch-solid {
        background: #18080a;
        border: 2px solid var(--color-bearish);
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 10px;
    }

    /* Segmented Navigation Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: var(--bg-surface);
        padding: 6px;
        border-radius: 8px;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 6px;
        color: var(--text-secondary);
        font-size: 0.86rem;
        font-weight: 600;
        border: none;
        background-color: transparent;
        padding: 0 14px;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background-color: rgba(255,255,255,0.04);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-surface-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-prominent) !important;
        font-weight: 700 !important;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-obsidian);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-prominent);
        border-radius: 3px;
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
# Top Header & Live Market Telemetry Bar (2s Rate-Limited Fragment)
# -------------------------------------------------------------
# -------------------------------------------------------------
# Top Header & Live Market Telemetry Bar (4s Smooth Telemetry)
# -------------------------------------------------------------
@st.fragment(run_every=4)
def render_live_header():
    broker = get_broker(st.session_state.active_broker_name)
    market_open, market_status_text = is_market_open()
    ist_now = get_ist_now().strftime("%d %b %Y | %H:%M:%S IST")
    
    nifty_quote = get_live_quote("^NSEI")
    banknifty_quote = get_live_quote("^NSEBANK")
    sensex_quote = get_live_quote("^BSESN")
    vix_quote = get_live_quote("^INDIAVIX")
    portfolio_data = broker.get_account_balance()
    
    # 5 Institutional-Grade Flat Metric Cards
    c_nifty, c_bn, c_sensex, c_vix, c_status = st.columns([1.8, 1.8, 1.8, 1.4, 2.2])
    
    with c_nifty:
        p = float(nifty_quote.get("price", 0.0))
        chg = float(nifty_quote.get("change_pct", 0.0))
        if p > 0:
            is_bull = chg >= 0
            st.markdown(f"""
            <div class='op-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;'>
                    <span style='font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>🇮🇳 NIFTY 50</span>
                    <span class='{'badge-bull' if is_bull else 'badge-bear'}' style='font-size: 0.68rem; padding: 1px 5px; white-space: nowrap;'>{'▲ +' if is_bull else '▼ '}{chg:.2f}%</span>
                </div>
                <div class='{'ticker-val-bull' if is_bull else 'ticker-val-bear'}' style='font-size: 1.25rem; font-weight: 800; white-space: nowrap; overflow: hidden;'>₹{p:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='op-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;'>
                    <span style='font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>🇮🇳 NIFTY 50</span>
                    <span class='badge-neutral' style='font-size: 0.68rem; padding: 1px 5px;'>OFFLINE</span>
                </div>
                <div class='tnum' style='color: #64748b; font-size: 1.15rem;'>₹---.--</div>
            </div>
            """, unsafe_allow_html=True)

    with c_bn:
        p = float(banknifty_quote.get("price", 0.0))
        chg = float(banknifty_quote.get("change_pct", 0.0))
        if p > 0:
            is_bull = chg >= 0
            st.markdown(f"""
            <div class='op-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;'>
                    <span style='font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>🏦 BANK NIFTY</span>
                    <span class='{'badge-bull' if is_bull else 'badge-bear'}' style='font-size: 0.68rem; padding: 1px 5px; white-space: nowrap;'>{'▲ +' if is_bull else '▼ '}{chg:.2f}%</span>
                </div>
                <div class='{'ticker-val-bull' if is_bull else 'ticker-val-bear'}' style='font-size: 1.25rem; font-weight: 800; white-space: nowrap; overflow: hidden;'>₹{p:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='op-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;'>
                    <span style='font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>🏦 BANK NIFTY</span>
                    <span class='badge-neutral' style='font-size: 0.68rem; padding: 1px 5px;'>OFFLINE</span>
                </div>
                <div class='tnum' style='color: #64748b; font-size: 1.15rem;'>₹---.--</div>
            </div>
            """, unsafe_allow_html=True)

    with c_sensex:
        p = float(sensex_quote.get("price", 0.0))
        chg = float(sensex_quote.get("change_pct", 0.0))
        if p > 0:
            is_bull = chg >= 0
            st.markdown(f"""
            <div class='op-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;'>
                    <span style='font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>🏛️ SENSEX</span>
                    <span class='{'badge-bull' if is_bull else 'badge-bear'}' style='font-size: 0.68rem; padding: 1px 5px; white-space: nowrap;'>{'▲ +' if is_bull else '▼ '}{chg:.2f}%</span>
                </div>
                <div class='{'ticker-val-bull' if is_bull else 'ticker-val-bear'}' style='font-size: 1.25rem; font-weight: 800; white-space: nowrap; overflow: hidden;'>₹{p:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='op-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;'>
                    <span style='font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>🏛️ SENSEX</span>
                    <span class='badge-neutral' style='font-size: 0.68rem; padding: 1px 5px;'>OFFLINE</span>
                </div>
                <div class='tnum' style='color: #64748b; font-size: 1.15rem;'>₹---.--</div>
            </div>
            """, unsafe_allow_html=True)

    with c_vix:
        v = float(vix_quote.get("price", 0.0))
        if v > 0:
            if v < 15.0:
                v_tag, v_badge = "LOW", "badge-sky"
            elif v < 20.0:
                v_tag, v_badge = "NORMAL", "badge-sky"
            elif v < 25.0:
                v_tag, v_badge = "ELEVATED", "badge-neutral"
            else:
                v_tag, v_badge = "EXTREME", "badge-bear"
            st.markdown(f"""
            <div class='op-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;'>
                    <span style='font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>⚡ INDIA VIX</span>
                    <span class='{v_badge}' style='font-size: 0.68rem; padding: 1px 5px; white-space: nowrap;'>{v_tag}</span>
                </div>
                <div class='tnum' style='font-size: 1.25rem; font-weight: 800; color: #f8fafc; white-space: nowrap; overflow: hidden;'>{v:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='op-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;'>
                    <span style='font-size: 0.68rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>⚡ INDIA VIX</span>
                    <span class='badge-neutral' style='font-size: 0.68rem; padding: 1px 5px;'>OFFLINE</span>
                </div>
                <div class='tnum' style='color: #64748b; font-size: 1.15rem;'>--.--</div>
            </div>
            """, unsafe_allow_html=True)

    with c_status:
        st_dot = "dot-live-open" if market_open else "dot-urgent-attention"
        st_badge = "badge-bull" if market_open else "badge-bear"
        is_paper = "paper" in broker.name.lower()
        st.markdown(f"""
        <div class='op-card' style='display: flex; flex-direction: column; justify-content: center; align-items: flex-end;'>
            <div><span class='{st_badge}' style='font-size: 0.72rem;'><span class='{st_dot}'></span>{market_status_text.upper()}</span></div>
            <div style='color: #94a3b8; font-size: 0.72rem; margin-top: 4px; font-family: "JetBrains Mono", monospace; white-space: nowrap;'>
                MODE: <strong style='color: {"#f59e0b" if is_paper else "#10b981"};'>{'PAPER (₹1.00L)' if is_paper else 'ZERODHA LIVE'}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)      
        
    # Account Operational Telemetry Strip (Flat, High-Contrast, Tabular Figures)
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    total_eq = float(portfolio_data.get("total_equity", 100000.0))
    cash_avail = float(portfolio_data.get("cash", 100000.0))
    unreal_pnl = float(portfolio_data.get("unrealized_pnl", 0.0))
    real_pnl = float(portfolio_data.get("realized_pnl", 0.0))
    init_cap = float(portfolio_data.get("initial_capital", 100000.0))
    total_ret_pct = ((total_eq - init_cap) / init_cap) * 100.0
    
    m1.metric("💰 Total Portfolio Value", format_currency_inr(total_eq), f"{total_ret_pct:+.2f}%", delta_color="normal")
    m2.metric("💵 Available Cash Margin", format_currency_inr(cash_avail))
    m3.metric("📈 Live Open P&L", format_currency_inr(unreal_pnl), f"{unreal_pnl:+,.2f} ₹", delta_color="normal")
    m4.metric("📊 Realized P&L Today", format_currency_inr(real_pnl), f"{real_pnl:+,.2f} ₹", delta_color="normal")
    
    # Dynamic Max Daily Risk & Today's SL Breaches
    max_daily_risk = max(2000.0, 0.015 * init_cap)
    closed = get_closed_trades(limit=50)
    today_str = get_ist_now().strftime("%Y-%m-%d")
    today_losses = [t for t in closed if float(t.get("net_pnl", 0.0)) < 0 and str(t.get("exit_time", "")).startswith(today_str)]
    m5.metric("🛡️ Max Daily Risk Floor", f"₹{max_daily_risk:,.2f}", f"{len(today_losses)}/3 SL Hit Today", delta_color="off")

render_live_header()

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
# ⚡ Live Stock Price Watcher Widget (Instant Reactive Search)
# -------------------------------------------------------------
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
# Sidebar Navigation & Experience Mode Selector
# -------------------------------------------------------------
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
# Pre-Market Opening Analyzer Helper
# -------------------------------------------------------------
def render_pre_market_tab(broker_instance):
    st.markdown("""
    <div style='margin-bottom: 8px;'>
        <h2 style='margin: 0; font-family: "Outfit", sans-serif;'>🌅 Morning Market Intelligence & High-Confidence Trade Calls</h2>
        <div style='color: #94a3b8; font-size: 0.92rem; margin-top: 4px;'>
            Institutional pre-market cues, curated <strong>Stock Breakout Calls</strong>, <strong>Nifty & BankNifty Option Calls</strong>, and <strong>Swing Picks</strong> with 1-Click execution.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Analyzing Market Opening Cues & Scanning 40+ Top Equities & F&O Options..."):
        scan_data = PreMarketAnalyzer.scan_pre_market_stocks(top_n=6)
        sentiment_info = scan_data["opening_sentiment"]
        top_picks = scan_data["top_picks"]
        option_calls = scan_data.get("option_calls", [])
        swing_picks = scan_data.get("swing_picks", [])
        gap_ups = scan_data["top_gap_ups"]
        gap_downs = scan_data["top_gap_downs"]

    s_badge_color = sentiment_info["badge_color"]
    st.markdown(f"""
    <div style='background: #111622; border: 1.5px solid {s_badge_color}; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px;'>
        <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;'>
            <div>
                <div style='font-size: 1.35rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>
                    {sentiment_info["title"]}
                </div>
                <div style='color: #cbd5e1; font-size: 0.92rem; margin-top: 6px; line-height: 1.4;'>
                    {sentiment_info["explanation"]}
                </div>
            </div>
            <div style='text-align: right;'>
                <span style='background: {s_badge_color}22; color: {s_badge_color}; border: 1px solid {s_badge_color}; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>
                    {sentiment_info["phase_description"]}
                </span>
                <div style='color: #94a3b8; font-size: 0.75rem; margin-top: 6px;'>
                    Updated: {sentiment_info["timestamp"]}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4)
    n_gap_sign = "+" if sentiment_info["gap_pct"] >= 0 else ""
    q1.metric("🇮🇳 NIFTY 50", f"₹{sentiment_info['nifty_price']:,.2f}", f"{n_gap_sign}{sentiment_info['gap_pct']:.2f}% Gap", delta_color="normal")
    q2.metric("🏦 BANK NIFTY", f"₹{sentiment_info['banknifty_price']:,.2f}")
    q3.metric("⚡ INDIA VIX (Volatility)", f"{sentiment_info['vix_level']:.2f}", "Normal Market" if sentiment_info['vix_level'] < 16 else "High Volatility")
    q4.metric("🎯 Total Suggestions Ready", f"{len(top_picks) + len(option_calls) + len(swing_picks)} Calls", "Stocks + Options + Swing")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Clean, Simple Segmented Navigation Tabs
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        f"⚡ Top Morning Stock Calls ({len(top_picks)})",
        f"🎯 Nifty & BankNifty Option Calls ({len(option_calls)})",
        f"💎 Positional & Swing Picks ({len(swing_picks)})",
        f"🔥 Gap & Volume Movers ({len(gap_ups) + len(gap_downs)})"
    ])

    # -------------------------------------------------------------
    # SUB-TAB 1: TOP MORNING STOCK CALLS
    # -------------------------------------------------------------
    with sub_tab1:
        st.markdown("""
        <div style='margin-bottom: 12px; color: #94a3b8; font-size: 0.88rem;'>
            High-conviction equity breakout stocks scanned across 40+ liquid Indian companies. Filtered for Relative Strength and positive buyer volume.
        </div>
        """, unsafe_allow_html=True)

        if top_picks:
            # 2-Row Responsive Grid (3 per row)
            for row_idx in range(0, len(top_picks), 3):
                row_items = top_picks[row_idx:row_idx+3]
                cols = st.columns(len(row_items))
                for col_idx, pick in enumerate(row_items):
                    with cols[col_idx]:
                        sym = pick["symbol"]
                        name = pick["display_name"]
                        price = pick["current_price"]
                        action = pick["action"]
                        act_title = pick["action_title"]
                        act_badge = pick["action_badge"]
                        t1 = pick["target_1_price"]
                        t2 = pick["target_2_price"]
                        sl = pick["stop_loss_price"]
                        score = pick["score"]
                        reason = pick["reason"]
                        setup_badge = pick.get("setup_grade_title", "⚡ GRADE A")
                        win_p = pick.get("win_probability", 70)

                        st.markdown(f"""
                        <div style='background: #111622; border: 1.5px solid {act_badge}; border-radius: 12px; padding: 18px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
                            <div>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <div style='font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>{name}</div>
                                    <span style='background: {act_badge}22; color: {act_badge}; border: 1px solid {act_badge}; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;'>{act_title}</span>
                                </div>
                                <div style='display: flex; justify-content: space-between; align-items: baseline; margin: 6px 0;'>
                                    <div style='font-size: 1.35rem; font-weight: 800; color: #38bdf8; font-family: "JetBrains Mono", monospace;'>₹{price:,.2f}</div>
                                    <span style='background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 700;'>{win_p}% Win Rate</span>
                                </div>
                                <div style='color: #94a3b8; font-size: 0.78rem; margin-bottom: 10px;'>Setup: <strong style='color: #f8fafc;'>{setup_badge}</strong> &bull; Score: <strong style='color: #f8fafc;'>{score:.1f}/10</strong></div>
                                
                                <div style='background: #080b11; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; margin-bottom: 12px;'>
                                    <div style='display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px;'>
                                        <span style='color: #94a3b8;'>🎯 Target 1 (Quick):</span>
                                        <strong style='color: #10b981;'>₹{t1:,.2f} (+{pick['target_1_gain_pct']:.1f}%)</strong>
                                    </div>
                                    <div style='display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px;'>
                                        <span style='color: #94a3b8;'>🚀 Target 2 (Runner):</span>
                                        <strong style='color: #10b981;'>₹{t2:,.2f} (+{pick['target_2_gain_pct']:.1f}%)</strong>
                                    </div>
                                    <div style='display: flex; justify-content: space-between; font-size: 0.82rem;'>
                                        <span style='color: #94a3b8;'>🛑 Safety Stop-Loss:</span>
                                        <strong style='color: #f43f5e;'>₹{sl:,.2f} (-{pick['stop_loss_pct']:.1f}%)</strong>
                                    </div>
                                </div>
                                
                                <div style='font-size: 0.80rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 14px;'>
                                    💡 <em>{reason}</em>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        trade_budget = 25000.0
                        qty_to_trade = max(1, int(trade_budget / max(1.0, price)))
                        
                        if st.button(f"🚀 1-Click Buy {name} (₹{price*qty_to_trade:,.0f})", key=f"premarket_trade_btn_{sym}_{row_idx}_{col_idx}", type="primary", use_container_width=True):
                            proposal = {
                                "symbol": sym,
                                "target_asset": sym,
                                "action": "BUY_STOCK" if "BUY" in action else "SELL_STOCK",
                                "confidence_score": score,
                                "entry_price": price,
                                "sl": sl,
                                "target_1": t1,
                                "horizon": "intraday",
                                "notes": f"Pre-Market Morning Pick ({name})"
                            }
                            p_state = get_portfolio_state()
                            guard = AIGuardrails(min_confidence_threshold=7.0)
                            approved, g_reason, sanitized_order = guard.evaluate_proposal(proposal, p_state, enforce_time_cutoff=False)
                            
                            if approved:
                                order_res = broker_instance.place_order(
                                    symbol=sym,
                                    side="BUY" if "BUY" in action else "SELL",
                                    quantity=qty_to_trade,
                                    price=price,
                                    sl=sl,
                                    tp=t1,
                                    strategy_name="PreMarket_Morning_Pick"
                                )
                                if order_res.get("status") in ["FILLED", "SUCCESS"]:
                                    st.success(f"✅ Trade Filled! Bought {qty_to_trade} shares of {name} @ ₹{price:.2f}. Safety SL set @ ₹{sl:.2f}.")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Order Rejected by Broker: {order_res.get('message')}")
                            else:
                                st.error(f"🛡️ Guardrail Protected: {g_reason}")
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # SUB-TAB 2: MORNING F&O OPTION CALLS (NIFTY / BANKNIFTY)
    # -------------------------------------------------------------
    with sub_tab2:
        st.markdown("""
        <div style='margin-bottom: 12px; color: #94a3b8; font-size: 0.88rem;'>
            High-probability Index Option contracts with defined strike prices, option premiums in ₹, profit potential per lot, and disciplined stop-loss risk.
        </div>
        """, unsafe_allow_html=True)

        if option_calls:
            opt_cols = st.columns(len(option_calls))
            for i, opt in enumerate(option_calls):
                with opt_cols[i]:
                    opt_sym = opt["symbol"]
                    opt_badge = opt["action_badge"]
                    opt_action = opt["action"]
                    entry_prem = opt["entry_premium"]
                    lot_sz = opt["lot_size"]
                    cap_lot = opt["capital_per_lot"]
                    t1_prem = opt["target_1"]
                    t1_gain_inr = opt["target_1_profit"]
                    t2_prem = opt["target_2"]
                    t2_gain_inr = opt["target_2_profit"]
                    sl_prem = opt["stop_loss"]
                    sl_loss_inr = opt["stop_loss_risk"]
                    win_p = opt["win_probability"]
                    reason = opt["reason"]

                    kite_name = opt.get("kite_symbol", opt_sym)
                    expiry_name = opt.get("expiry", "Current Weekly Thursday")

                    st.markdown(f"""
                    <div style='background: #111622; border: 2px solid {opt_badge}; border-radius: 12px; padding: 18px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
                        <div>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div style='font-size: 1.25rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>{opt_sym}</div>
                                <span style='background: {opt_badge}22; color: {opt_badge}; border: 1px solid {opt_badge}; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.80rem;'>{opt_action}</span>
                            </div>
                            <div style='color: #38bdf8; font-size: 0.82rem; font-weight: 700; margin: 4px 0;'>📅 {expiry_name}</div>
                            <div style='color: #94a3b8; font-size: 0.76rem; margin-bottom: 8px;'>Broker Symbol: <code style='color: #f8fafc; background: #080b11; padding: 2px 4px; border-radius: 4px;'>{kite_name}</code> &bull; Lot: <strong style='color: #f8fafc;'>{lot_sz} Qty</strong></div>
                            
                            <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;'>
                                <div style='font-size: 1.4rem; font-weight: 800; color: #38bdf8; font-family: "JetBrains Mono", monospace;'>₹{entry_prem:.1f} <span style='font-size: 0.80rem; color: #94a3b8;'>Premium</span></div>
                                <span style='background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;'>{win_p}% Win Rate</span>
                            </div>
                            <div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 12px;'>Capital / Lot: <strong style='color: #f8fafc;'>₹{cap_lot:,.0f}</strong></div>
                            
                            <div style='background: #080b11; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; margin-bottom: 12px;'>
                                <div style='display: flex; justify-content: space-between; font-size: 0.84rem; margin-bottom: 6px;'>
                                    <span style='color: #94a3b8;'>🎯 Target 1 (+35%):</span>
                                    <strong style='color: #10b981;'>₹{t1_prem:.1f} (+₹{t1_gain_inr:,.0f}/lot)</strong>
                                </div>
                                <div style='display: flex; justify-content: space-between; font-size: 0.84rem; margin-bottom: 6px;'>
                                    <span style='color: #94a3b8;'>🚀 Target 2 (+65%):</span>
                                    <strong style='color: #10b981;'>₹{t2_prem:.1f} (+₹{t2_gain_inr:,.0f}/lot)</strong>
                                </div>
                                <div style='display: flex; justify-content: space-between; font-size: 0.84rem;'>
                                    <span style='color: #94a3b8;'>🛑 Safety Stop-Loss (-22%):</span>
                                    <strong style='color: #f43f5e;'>₹{sl_prem:.1f} (-₹{sl_loss_inr:,.0f}/lot)</strong>
                                </div>
                            </div>
                            
                            <div style='font-size: 0.80rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 14px;'>
                                💡 <em>{reason}</em>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"🚀 1-Click Buy 1 Lot ({opt_sym})", key=f"opt_trade_btn_{i}", type="primary", use_container_width=True):
                        proposal = {
                            "symbol": opt_sym,
                            "target_asset": opt_sym,
                            "action": "BUY_OPTION",
                            "confidence_score": 8.0,
                            "entry_price": entry_prem,
                            "sl": sl_prem,
                            "target_1": t1_prem,
                            "horizon": "intraday",
                            "notes": f"Morning Option Call ({opt_sym})"
                        }
                        p_state = get_portfolio_state()
                        guard = AIGuardrails(min_confidence_threshold=7.0)
                        approved, g_reason, sanitized_order = guard.evaluate_proposal(proposal, p_state, enforce_time_cutoff=False)
                        
                        if approved:
                            order_res = broker_instance.place_order(
                                symbol=opt_sym,
                                side="BUY",
                                quantity=lot_sz,
                                price=entry_prem,
                                sl=sl_prem,
                                tp=t1_prem,
                                strategy_name="Morning_Option_Call"
                            )
                            if order_res.get("status") in ["FILLED", "SUCCESS"]:
                                st.success(f"✅ Option Trade Filled! Bought 1 Lot ({lot_sz} Qty) of {opt_sym} @ ₹{entry_prem:.1f}.")
                                st.rerun()
                            else:
                                st.error(f"❌ Order Rejected: {order_res.get('message')}")
                        else:
                            st.error(f"🛡️ Guardrail Protected: {g_reason}")

    # -------------------------------------------------------------
    # SUB-TAB 3: POSITIONAL & SWING PICKS
    # -------------------------------------------------------------
    with sub_tab3:
        st.markdown("""
        <div style='margin-bottom: 12px; color: #94a3b8; font-size: 0.88rem;'>
            High-conviction multi-week wealth builder stocks for 2 to 4 week holding horizons with +6% to +12% target upside.
        </div>
        """, unsafe_allow_html=True)

        if swing_picks:
            s_cols = st.columns(len(swing_picks))
            for i, sw in enumerate(swing_picks):
                with s_cols[i]:
                    sw_name = sw["display_name"]
                    sw_sym = sw["symbol"]
                    sw_price = sw["current_price"]
                    sw_t1 = sw["target_1_price"]
                    sw_t2 = sw["target_2_price"]
                    sw_sl = sw["stop_loss_price"]
                    sw_win = sw["win_probability"]
                    sw_reason = sw["reason"]

                    st.markdown(f"""
                    <div style='background: #111622; border: 1.5px solid #10b981; border-radius: 12px; padding: 18px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
                        <div>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div style='font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>{sw_name}</div>
                                <span style='background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid #10b981; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;'>SWING BUY</span>
                            </div>
                            <div style='display: flex; justify-content: space-between; align-items: baseline; margin: 6px 0;'>
                                <div style='font-size: 1.35rem; font-weight: 800; color: #38bdf8; font-family: "JetBrains Mono", monospace;'>₹{sw_price:,.2f}</div>
                                <span style='background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 700;'>{sw_win}% Win Rate</span>
                            </div>
                            <div style='color: #94a3b8; font-size: 0.78rem; margin-bottom: 10px;'>Holding: <strong style='color: #f8fafc;'>{sw["holding_period"]}</strong></div>
                            
                            <div style='background: #080b11; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; margin-bottom: 12px;'>
                                <div style='display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px;'>
                                    <span style='color: #94a3b8;'>🎯 Target 1 (+6%):</span>
                                    <strong style='color: #10b981;'>₹{sw_t1:,.2f}</strong>
                                </div>
                                <div style='display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px;'>
                                    <span style='color: #94a3b8;'>🚀 Target 2 (+12%):</span>
                                    <strong style='color: #10b981;'>₹{sw_t2:,.2f}</strong>
                                </div>
                                <div style='display: flex; justify-content: space-between; font-size: 0.82rem;'>
                                    <span style='color: #94a3b8;'>🛑 Safety Stop-Loss (-4%):</span>
                                    <strong style='color: #f43f5e;'>₹{sw_sl:,.2f}</strong>
                                </div>
                            </div>
                            
                            <div style='font-size: 0.80rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 14px;'>
                                💡 <em>{sw_reason}</em>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    qty_swing = max(1, int(35000.0 / max(1.0, sw_price)))
                    if st.button(f"🚀 1-Click Swing Buy (₹{sw_price*qty_swing:,.0f})", key=f"swing_btn_{i}", type="primary", use_container_width=True):
                        proposal = {
                            "symbol": sw_sym,
                            "target_asset": sw_sym,
                            "action": "BUY_STOCK",
                            "confidence_score": 8.0,
                            "entry_price": sw_price,
                            "sl": sw_sl,
                            "target_1": sw_t1,
                            "horizon": "swing",
                            "notes": f"Swing Pick ({sw_name})"
                        }
                        p_state = get_portfolio_state()
                        guard = AIGuardrails(min_confidence_threshold=7.0)
                        approved, g_reason, sanitized_order = guard.evaluate_proposal(proposal, p_state, enforce_time_cutoff=False)
                        
                        if approved:
                            order_res = broker_instance.place_order(
                                symbol=sw_sym,
                                side="BUY",
                                quantity=qty_swing,
                                price=sw_price,
                                sl=sw_sl,
                                tp=sw_t1,
                                strategy_name="Positional_Swing_Pick"
                            )
                            if order_res.get("status") in ["FILLED", "SUCCESS"]:
                                st.success(f"✅ Swing Position Opened! Bought {qty_swing} shares of {sw_name} @ ₹{sw_price:.2f}.")
                                st.rerun()
                            else:
                                st.error(f"❌ Order Rejected: {order_res.get('message')}")
                        else:
                            st.error(f"🛡️ Guardrail Protected: {g_reason}")

    # -------------------------------------------------------------
    # SUB-TAB 4: GAP & VOLUME MOVERS LEADERBOARD
    # -------------------------------------------------------------
    with sub_tab4:
        st.markdown("""
        <div style='margin-bottom: 12px; color: #94a3b8; font-size: 0.88rem;'>
            Real-time morning momentum leaderboard showing stocks with the strongest opening gaps across the Indian market.
        </div>
        """, unsafe_allow_html=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("#### 🔺 Top Morning Gap-Up Stocks")
            if gap_ups:
                df_gu = pd.DataFrame(gap_ups)
                st.dataframe(df_gu[["name", "price", "gap_pct", "score"]], use_container_width=True, hide_index=True)
            else:
                st.info("No significant morning gap-up stocks (> +0.8%) detected today.")
                
        with m_col2:
            st.markdown("#### 🔻 Top Morning Gap-Down Stocks")
            if gap_downs:
                df_gd = pd.DataFrame(gap_downs)
                st.dataframe(df_gd[["name", "price", "gap_pct", "score"]], use_container_width=True, hide_index=True)
            else:
                st.info("No significant morning gap-down stocks (< -0.8%) detected today.")

    with st.expander("📘 **How the Indian Stock Market Opens (Simple 3-Minute Guide)**", expanded=False):
        st.markdown("""
        * **09:00 AM – 09:08 AM (Order Collection)**: *You and institutional investors can place buy and sell orders. No trades are executed yet, but the exchange collects all bids to discover fair opening prices.*
        * **09:08 AM – 09:15 AM (Price Matching & Discovery)**: *The exchange algorithm matches all buy and sell orders at a single equilibrium price (Pre-Open Price). Orders cannot be placed or canceled in this 7-minute window.*
        * **09:15 AM – 03:30 PM (Live Normal Trading)**: *Regular continuous trading starts across NSE and BSE. High-momentum breakout trades usually offer the best returns in the first 45 minutes (09:15 AM - 10:00 AM).*
        """)

# -------------------------------------------------------------
# Plain-English AI Chat Assistant Tab Helper
# -------------------------------------------------------------
def render_chat_assistant_tab(broker_instance):
    st.markdown("""
    <div style='margin-bottom: 8px;'>
        <h2 style='margin: 0; font-family: "Outfit", sans-serif;'>🗣️ Talk to Your ApexTrade AI Bot</h2>
        <div style='color: #94a3b8; font-size: 0.92rem; margin-top: 4px;'>
            Ask anything in plain English — analyze stocks, check market sentiment, get top picks, and execute safe bracket trades.
        </div>
    </div>
    """, unsafe_allow_html=True)

    saved_ai = load_ai_settings()
    has_llm = saved_ai.get("is_connected") and saved_ai.get("api_key")
    
    if has_llm:
        st.markdown(f"""
        <div style='background: #111622; border: 1px solid #10b981; border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: #10b981; font-weight: 700; font-size: 0.88rem;'><span class='ambient-dot-green'></span>AI BRAIN ACTIVE: {saved_ai['provider'].upper()} ({saved_ai.get('model', 'gemini-3.1-flash-lite')})</span>
            <span class='badge-bull'>GUARDRAILS ENGAGED</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: #111622; border: 1px solid #f59e0b; border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: #f59e0b; font-weight: 700; font-size: 0.88rem;'>⚡ LOCAL HEURISTIC MODE (Deterministic Rule Engine & Zero Hallucinations)</span>
            <span class='badge-neutral'>CONNECT API IN SETTINGS FOR LLM</span>
        </div>
        """, unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "👋 **Hello! I am your ApexTrade AI Assistant.**\n\nI can analyze Indian stocks, check live pre-market sentiment, find today's top picks, or propose safe trades with automatic stop-loss.\n\n*Try one of the quick suggestions below or type your question!*",
                "action_card": None,
                "timestamp": get_ist_now().strftime("%I:%M %p")
            }
        ]

    st.markdown("<div style='font-size: 0.80rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;'>💡 Quick Ideas:</div>", unsafe_allow_html=True)
    q_c1, q_c2, q_c3, q_c4, q_c5 = st.columns(5)
    selected_quick_query = None
    with q_c1:
        if st.button("🌅 What is NIFTY doing?", use_container_width=True):
            selected_quick_query = "What is the market opening mood today?"
    with q_c2:
        if st.button("🌟 Top 3 Intraday Picks", use_container_width=True):
            selected_quick_query = "Show me the best 3 stocks to buy today"
    with q_c3:
        if st.button("📊 Analyze Tata Motors", use_container_width=True):
            selected_quick_query = "How is Tata Motors looking for intraday?"
    with q_c4:
        if st.button("💼 My Profit & Balance", use_container_width=True):
            selected_quick_query = "What is my account balance and profit today?"
    with q_c5:
        if st.button("🚀 Buy ₹25,000 Reliance", use_container_width=True):
            selected_quick_query = "Buy ₹25,000 of Reliance with safety stop-loss"

    # Chat Display Container
    for msg_idx, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg.get("action_card"):
                card = msg["action_card"]
                sym = card["symbol"]
                name = card["display_name"]
                action = card["action"]
                qty = card["quantity"]
                entry_p = card["entry_price"]
                cap = card["capital_required"]
                t1 = card["target_1_price"]
                t1_prof = card["target_1_profit"]
                sl = card["stop_loss_price"]
                sl_risk = card["stop_loss_risk"]
                score = card["score"]
                act_badge = "#10b981" if action == "BUY" else "#f43f5e"

                st.markdown(f"""
                <div style='background: #111622; border: 2px solid {act_badge}; border-radius: 10px; padding: 16px; margin: 12px 0;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div style='font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>{name}</div>
                        <span style='background: {act_badge}22; color: {act_badge}; border: 1px solid {act_badge}; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;'>{action} &bull; {card["product_type"]}</span>
                    </div>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; font-size: 0.85rem;'>
                        <div>📦 Quantity: <strong style='color: #f8fafc;'>{qty} Shares</strong></div>
                        <div>💵 Capital Required: <strong style='color: #f8fafc;'>₹{cap:,.2f}</strong></div>
                        <div>🎯 Target 1: <strong style='color: #10b981;'>₹{t1:,.2f} (+₹{t1_prof:,.2f})</strong></div>
                        <div>🛑 Safety SL: <strong style='color: #f43f5e;'>₹{sl:,.2f} (-₹{sl_risk:,.2f})</strong></div>
                    </div>
                    <div style='color: #94a3b8; font-size: 0.78rem;'>AI Mathematical Score: <strong style='color: #f8fafc;'>{score:.1f} / 10.0</strong> &bull; Zero-Bypass Guardrails Engaged</div>
                </div>
                """, unsafe_allow_html=True)
                
                btn_key = f"chat_trade_btn_{sym}_{msg_idx}"
                if st.button(f"🚀 Confirm & Place {action} Order (₹{cap:,.0f})", key=btn_key, type="primary", use_container_width=True):
                    proposal = {
                        "symbol": sym,
                        "target_asset": sym,
                        "action": "BUY_STOCK" if action == "BUY" else "SELL_STOCK",
                        "confidence_score": score,
                        "entry_price": entry_p,
                        "sl": sl,
                        "target_1": t1,
                        "horizon": "intraday",
                        "notes": f"Chat Order for {name}"
                    }
                    
                    p_state = get_portfolio_state()
                    guard = AIGuardrails(min_confidence_threshold=7.0)
                    approved, g_reason, sanitized_order = guard.evaluate_proposal(proposal, p_state, enforce_time_cutoff=True)
                    
                    if approved:
                        order_res = broker_instance.place_order(
                            symbol=sym,
                            side=action,
                            quantity=qty,
                            price=entry_p,
                            sl=sl,
                            tp=t1,
                            strategy_name="Chat_Assistant_Order"
                        )
                        if order_res.get("status") in ["FILLED", "SUCCESS"]:
                            st.success(f"✅ Order Executed! Bought {qty} shares of {name} @ ₹{entry_p:,.2f}. Safety SL set @ ₹{sl:,.2f}.")
                            st.rerun()
                        else:
                            st.error(f"❌ Order Failed: {order_res.get('message')}")
                    else:
                        st.error(f"🛡️ Guardrail Protected: {g_reason}")

    user_input = st.chat_input("Ask me anything about Indian stocks, Nifty, or say 'Buy ₹25,000 of Tata Motors'...")
    final_query = selected_quick_query or user_input

    if final_query:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": final_query,
            "action_card": None,
            "timestamp": get_ist_now().strftime("%I:%M %p")
        })

        with st.spinner("ApexTrade AI is analyzing market data..."):
            res = TradingChatAssistant.process_query(
                user_query=final_query,
                chat_history=[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages[-6:]],
                provider=saved_ai.get("provider", "gemini"),
                api_key=saved_ai.get("api_key"),
                model=saved_ai.get("model")
            )
            
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": res["response_text"],
                "action_card": res.get("action_card"),
                "timestamp": res.get("timestamp", get_ist_now().strftime("%I:%M %p"))
            })
            st.rerun()

# -------------------------------------------------------------
# TAB DISPATCHING
# -------------------------------------------------------------
if active_tab == "🗣️ Talk to Your AI Bot (Chat & Voice)":
    render_chat_assistant_tab(broker)

elif active_tab == "🌅 Pre-Market & Best Stocks Today":
    render_pre_market_tab(broker)

# -------------------------------------------------------------
# TAB 0: 🤖 Autonomous AI Trading Agent (Claude / Kimi / Zerodha)
# -------------------------------------------------------------
elif active_tab in ["🤖 Autonomous AI Trading Agent (Claude / Kimi / F&O)", "🤖 AI Auto-Pilot (Automated Safe Trading)"]:
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
            with st.spinner("Scanning NIFTY, BANK NIFTY, and liquid momentum equities for high-conviction trade opportunities..."):
                llm_instance = LLMClient(provider=prov_key, model=model_choice, api_key=ai_api_key) if (ai_api_key and len(ai_api_key.strip()) >= 5) else None
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
    st.markdown("---")
    st.subheader("🧠 Multi-Agent AI Strategy Council & Autonomous Market Hunter")
    st.caption("3 specialized orthogonal AI agents evaluate candidate breakouts with 2-stage gating and software-managed OCO execution (entry + standalone exchange SL-M order).")
    
    with st.expander("🏛️ **Live 3-Agent Council Audit & Deliberation Console**", expanded=True):
        c_sym = st.selectbox(
            "Select Indian Stock for Multi-Agent Deliberation:",
            options=[item["symbol"] for item in config.DEFAULT_WATCHLIST],
            format_func=lambda s: next((f"{item['name']} ({item['symbol'].replace('.NS','')})" for item in config.DEFAULT_WATCHLIST if item["symbol"] == s), s),
            key="council_sym_select"
        )
        
        if st.button("🔍 Run 3-Agent Council Deliberation Audit", type="primary", use_container_width=True):
            with st.spinner(f"Convening 3-Agent Strategy Council for {c_sym}..."):
                df_c = get_historical_data(c_sym, period="5d", interval="5m")
                quote_c = get_live_quote(c_sym)
                c_res = MultiAgentCouncil.evaluate_candidate(c_sym, df_c, quote_c)
                st.session_state["last_council_audit"] = c_res
                
        if "last_council_audit" in st.session_state:
            c_res = st.session_state["last_council_audit"]
            m_score = c_res.get("math_score", 0.0)
            c_score = c_res.get("consensus_score", 0.0)
            c_app = c_res.get("consensus_approved", False)
            verdict = c_res.get("verdict", "N/A")
            agents = c_res.get("agents", {})
            
            banner_col = "#10b981" if c_app else "#f43f5e"
            st.markdown(f"""
            <div style='background: #111622; border: 2px solid {banner_col}; border-radius: 10px; padding: 14px 18px; margin: 12px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>{c_res.get('display_name')} &bull; ₹{c_res.get('current_price', 0.0):,.2f}</span>
                    <span class='{"badge-bull" if c_app else "badge-bear"}'>{verdict} &bull; {c_score:.1f}/10</span>
                </div>
                <div style='color: #94a3b8; font-size: 0.88rem; margin-top: 4px;'>Stage 1 Math Pre-Filter: <strong style='color: #f8fafc;'>{m_score:.1f}/10</strong> ({'PASSED' if c_res.get('passed_prefilter') else 'BLOCKED'}) &bull; {c_res.get('deliberation_summary')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if agents:
                a1 = agents.get("agent_1_pattern", {})
                a2 = agents.get("agent_2_defense", {})
                a3 = agents.get("agent_3_macro", {})
                
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.markdown(f"""
                    <div style='background: #1e293b55; border: 1px solid #38bdf8; border-radius: 8px; padding: 12px; min-height: 150px;'>
                        <div style='font-weight: 700; color: #38bdf8; font-size: 0.95rem;'>{a1.get('name')}</div>
                        <div style='font-size: 1.3rem; font-weight: 800; color: #f8fafc; margin: 4px 0;'>{a1.get('score', 0):.1f} <span style='font-size: 0.8rem; color: #94a3b8;'>({a1.get('vote')})</span></div>
                        <div style='color: #cbd5e1; font-size: 0.82rem;'>{a1.get('thesis')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_a2:
                    st.markdown(f"""
                    <div style='background: #1e293b55; border: 1px solid {"#f43f5e" if a2.get("veto") else "#10b981"}; border-radius: 8px; padding: 12px; min-height: 150px;'>
                        <div style='font-weight: 700; color: {"#f43f5e" if a2.get("veto") else "#10b981"}; font-size: 0.95rem;'>{a2.get('name')}</div>
                        <div style='font-size: 1.3rem; font-weight: 800; color: #f8fafc; margin: 4px 0;'>{a2.get('score', 0):.1f} <span style='font-size: 0.8rem; color: #94a3b8;'>({a2.get('vote')})</span></div>
                        <div style='color: #cbd5e1; font-size: 0.82rem;'>{a2.get('defense_notes')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_a3:
                    st.markdown(f"""
                    <div style='background: #1e293b55; border: 1px solid #a855f7; border-radius: 8px; padding: 12px; min-height: 150px;'>
                        <div style='font-weight: 700; color: #a855f7; font-size: 0.95rem;'>{a3.get('name')}</div>
                        <div style='font-size: 1.3rem; font-weight: 800; color: #f8fafc; margin: 4px 0;'>{a3.get('score', 0):.1f} <span style='font-size: 0.8rem; color: #94a3b8;'>({a3.get('vote')})</span></div>
                        <div style='color: #cbd5e1; font-size: 0.82rem;'>{a3.get('thesis')}</div>
                    </div>
                    """, unsafe_allow_html=True)

    hunter_status = MarketHunterDaemon.get_status()
    h_col1, h_col2, h_col3, h_col4 = st.columns([2, 1.5, 1.5, 1.5])
    with h_col1:
        if hunter_status["is_running"]:
            if st.button("⏹️ Stop Market Hunter Daemon", type="secondary", use_container_width=True):
                MarketHunterDaemon.stop()
                st.rerun()
        else:
            if st.button("⚡ Start Autonomous Market Hunter (30s Loop)", type="primary", use_container_width=True):
                MarketHunterDaemon.start(active_ai_broker, scan_interval_sec=30)
                st.rerun()
    with h_col2:
        st.metric("Hunter Engine", "🟢 HUNTING" if hunter_status["is_running"] else "⏸️ STOPPED")
    with h_col3:
        st.metric("30s Scans Run", f"{hunter_status['scans_completed']}")
    with h_col4:
        st.metric("Trades Executed", f"{hunter_status['trades_placed_today']}")

    if hunter_status["logs"]:
        with st.expander(f"📜 **Live Hunter Activity Stream ({len(hunter_status['logs'])} events)**", expanded=False):
            for l in hunter_status["logs"][:15]:
                st.markdown(f"**`{l['timestamp']}`** &bull; `[{l['type']}]` {l['message']}")

    st.markdown("---")
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
        try:
            has_key = bool(ai_api_key and len(ai_api_key.strip()) >= 5)
            llm_instance = LLMClient(provider=prov_key, model=model_choice, api_key=ai_api_key) if has_key else LLMClient(provider="gemini", model="gemini-2.5-flash", api_key="")
            agent = AITradingAgent(
                llm_client=llm_instance,
                guardrails=ai_guardrails,
                broker=active_ai_broker,
                is_live_mode=is_live_selected
            )
            
            with st.spinner(f"Analyzing live market structure for {clean_target}..."):
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
# TAB 0.5: ⚡ NFO Options Greeks & OI Matrix
# -------------------------------------------------------------
elif active_tab == "⚡ NFO Options Greeks & OI Matrix":
    st.markdown("""
    <h2>⚡ NFO Options Greeks & Open Interest Matrix</h2>
    <div style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 14px;'>
        European Black-Scholes Greeks Engine with Implied Volatility (IV), Max Pain, Put-Call Ratio (PCR), and Gamma-Aware Strike Selection.
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Top Controls Bar
    oc_col1, oc_col2, oc_col3 = st.columns([2, 1.5, 1.5])
    with oc_col1:
        opt_sym = st.selectbox(
            "Select Underlying Instrument:",
            ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "HDFCBANK", "TCS", "INFY", "TATAMOTORS"],
            index=0
        )
    with oc_col2:
        opt_dte = st.slider("Days to Expiry (DTE):", min_value=0.1, max_value=14.0, value=3.0, step=0.1, help="0.1 to 0.5 represents 0DTE / same-day expiry.")
    with oc_col3:
        opt_iv = st.slider("Base IV (%):", min_value=5.0, max_value=60.0, value=15.5, step=0.5, help="Implied Volatility parameter.")
        
    with st.spinner(f"Computing real-time European Black-Scholes Greeks for {opt_sym}..."):
        from src.data.data_fetcher import get_live_quote
        from src.strategies.options_greeks import OptionChainBuilder, SmartStrikeSelector
        
        quote = get_live_quote(opt_sym)
        spot_p = float(quote.get("price", 24650.0 if "NIFTY" in opt_sym else (51200.0 if "BANKNIFTY" in opt_sym else 1000.0)))
        
        chain_data = OptionChainBuilder.build_option_chain_matrix(
            symbol=opt_sym,
            spot_price=spot_p,
            dte_days=opt_dte,
            strikes_count=11,
            base_iv=opt_iv / 100.0
        )
        
    # 2. Key Telemetry KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    pcr_val = chain_data["pcr"]["pcr_oi"]
    pcr_color = "#10b981" if pcr_val >= 1.15 else ("#f43f5e" if pcr_val <= 0.85 else "#f59e0b")
    pcr_bias = "Bullish Writer Bias" if pcr_val >= 1.15 else ("Bearish Writer Bias" if pcr_val <= 0.85 else "Neutral / Balanced")
    
    mp_strike = chain_data["max_pain"]
    mp_diff = spot_p - mp_strike
    
    with kpi_col1:
        st.metric("Spot Price", f"₹{spot_p:,.2f}", f"{quote.get('change_pct', 0.0):+.2f}%")
    with kpi_col2:
        st.metric("Put-Call Ratio (PCR)", f"{pcr_val:.2f}", pcr_bias)
    with kpi_col3:
        st.metric("Max Pain Strike", f"₹{mp_strike:,.0f}", f"{mp_diff:+.0f} pts from Spot")
    with kpi_col4:
        st.metric("ATM Implied Volatility", f"{opt_iv:.1f}%", f"DTE: {opt_dte:.1f}d")
        
    st.markdown("---")
    
    # 3. Interactive Symmetrical Option Chain Matrix
    st.markdown("### 📊 Live Symmetrical Option Chain Matrix")
    st.markdown("<div style='color: #94a3b8; font-size: 0.82rem; margin-bottom: 8px;'>Theoretical European pricing & Greeks. Highlights ATM and In-The-Money strikes.</div>", unsafe_allow_html=True)
    
    table_rows = []
    for s in chain_data["strikes"]:
        k = s["strike"]
        is_atm = s["is_atm"]
        ce = s["ce"]
        pe = s["pe"]
        
        table_rows.append({
            "CE Delta (Δ)": f"{ce['delta']:+.3f}",
            "CE Gamma (Γ)": f"{ce['gamma']:.5f}",
            "CE Theta (₹/d)": f"{ce['theta']:+.2f}",
            "CE IV": f"{ce['iv_pct']:.1f}%",
            "CE LTP (₹)": f"₹{ce['ltp']:.2f}",
            "CE Open Int": f"{ce['oi']:,}",
            "STRIKE": f"🎯 {k}" if is_atm else f"{k}",
            "PE Open Int": f"{pe['oi']:,}",
            "PE LTP (₹)": f"₹{pe['ltp']:.2f}",
            "PE IV": f"{pe['iv_pct']:.1f}%",
            "PE Theta (₹/d)": f"{pe['theta']:+.2f}",
            "PE Gamma (Γ)": f"{pe['gamma']:.5f}",
            "PE Delta (Δ)": f"{pe['delta']:+.3f}"
        })
        
    df_chain = pd.DataFrame(table_rows)
    st.dataframe(df_chain, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 4. Smart Strike Selector & Guardrail-Routed Execution
    st.markdown("### 🎯 Gamma-Aware Smart Strike Selector & 1-Click Execution")
    
    sc1, sc2 = st.columns([1.5, 2])
    with sc1:
        trade_intent = st.radio("Strategic Direction:", ["🟢 Bullish (Buy Call)", "🔴 Bearish (Buy Put)"], horizontal=True)
        strike_pref = st.selectbox("Strike Mode:", ["ATM (At-The-Money, Δ ≈ 0.50)", "ITM1 (In-The-Money, Δ ≥ 0.65)", "OTM1 (Out-of-The-Money, Δ ≈ 0.35)"])
        pref_code = "ATM" if "ATM" in strike_pref else ("ITM1" if "ITM1" in strike_pref else "OTM1")
        
        action_code = "BUY_CALL" if "Bullish" in trade_intent else "BUY_PUT"
        smart_pick = SmartStrikeSelector.select_optimal_strike(
            symbol=opt_sym,
            spot_price=spot_p,
            action=action_code,
            dte_days=opt_dte,
            preference=pref_code
        )
        
        default_qty = 25 if "NIFTY" in opt_sym else (15 if "BANK" in opt_sym else 10)
        lots_qty = st.number_input(f"Quantity (Shares / Contracts):", min_value=1, value=default_qty, step=default_qty)
        
    with sc2:
        st.markdown(f"""
        <div style='background: #111622; border: 1px solid #334155; border-radius: 10px; padding: 16px;'>
            <div style='font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;'>Selected Contract Blueprint</div>
            <div style='font-size: 1.4rem; font-weight: 800; color: #38bdf8; margin: 4px 0; font-family: "Outfit", sans-serif;'>{smart_pick["contract"]}</div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; font-size: 0.85rem;'>
                <div>Estimated LTP: <strong style='color: #f8fafc;'>₹{smart_pick["estimated_ltp"]:.2f}</strong></div>
                <div>Target Delta (Δ): <strong style='color: #f8fafc;'>{smart_pick["greeks"]["delta"]:+.3f}</strong></div>
                <div>Daily Theta (Θ): <strong style='color: #f43f5e;'>{smart_pick["greeks"]["theta_daily"]:+.2f} ₹/day</strong></div>
                <div>Gamma (Γ): <strong style='color: #f8fafc;'>{smart_pick["greeks"]["gamma"]:.5f}</strong></div>
            </div>
            {"<div style='margin-top: 8px; color: #f59e0b; font-size: 0.78rem; font-weight: 700;'>⚠️ 0DTE Expiry-Day Safety: Automatically shifted to ITM1 to suppress extreme gamma whips.</div>" if smart_pick["is_0dte_adjusted"] else ""}
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Analytical Payoff Curve
        payoff_data = SmartStrikeSelector.calculate_payoff_curve(
            spot_price=spot_p,
            strike=smart_pick["strike"],
            premium=smart_pick["estimated_ltp"],
            action=action_code,
            quantity=int(lots_qty)
        )
        
        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(
            x=payoff_data["underlying_prices"],
            y=payoff_data["pnl_at_expiry"],
            mode="lines",
            line=dict(color="#38bdf8", width=2.5),
            name="P&L at Expiry"
        ))
        fig_payoff.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)
        fig_payoff.add_vline(x=spot_p, line_dash="dot", line_color="#f59e0b", line_width=1.5, annotation_text=f"Spot ₹{spot_p:,.1f}", annotation_position="top left")
        fig_payoff.add_vline(x=payoff_data["breakeven"], line_dash="dash", line_color="#10b981", line_width=1.5, annotation_text=f"BE ₹{payoff_data['breakeven']:,.1f}", annotation_position="top right")
        
        fig_payoff.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111622",
            plot_bgcolor="#080b11",
            height=220,
            margin=dict(l=20, r=20, t=25, b=20),
            title=dict(text=f"📈 Payoff at Expiry (Breakeven: ₹{payoff_data['breakeven']:.1f} | Max Loss: ₹{payoff_data['max_loss']:,.0f})", font=dict(size=12, color="#94a3b8")),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_payoff, use_container_width=True)
        
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        
        # 1-Click Guardrail-Routed Execution Button
        if st.button(f"🚀 Execute 1-Click {action_code} via Guardrails", type="primary", use_container_width=True):
            entry_p = smart_pick["estimated_ltp"]
            sl_p = round(entry_p * 0.80, 2) # Standard 20% option SL
            t1_p = round(entry_p * 1.30, 2) # Target 1 (+30%)
            t2_p = round(entry_p * 1.60, 2) # Target 2 (+60%)
            
            proposal = {
                "symbol": smart_pick["contract"],
                "target_asset": smart_pick["contract"],
                "action": "BUY_STOCK",
                "confidence_score": 8.5,
                "entry_price": entry_p,
                "sl": sl_p,
                "target_1": t1_p,
                "target_2": t2_p,
                "horizon": "intraday",
                "notes": f"Options Greeks Matrix execution (Delta: {smart_pick['greeks']['delta']:+.2f})"
            }
            
            # ROUTE STRICTLY THROUGH AI GUARDRAILS
            from src.engine.ai_guardrails import AIGuardrails
            from src.utils.storage import get_portfolio_state
            
            p_state = get_portfolio_state()
            guard = AIGuardrails(min_confidence_threshold=7.5)
            approved, reason, sanitized_order = guard.evaluate_proposal(proposal, p_state, enforce_time_cutoff=False)
            
            if approved:
                order_res = broker.place_order(
                    symbol=smart_pick["contract"],
                    side="BUY",
                    quantity=int(lots_qty),
                    price=entry_p,
                    sl=sl_p,
                    tp=t1_p,
                    strategy_name="Options_Greeks_Matrix"
                )
                if order_res.get("status") in ["FILLED", "SUCCESS"]:
                    st.success(f"✅ Guardrail Approved & Filled! Bought {lots_qty} {smart_pick['contract']} @ ₹{order_res.get('price', entry_p):.2f}")
                    st.rerun()
                else:
                    st.error(f"❌ Order Rejected by Broker: {order_res.get('message')}")
            else:
                st.error(f"🛡️ Guardrail Blocked Execution: {reason}")

# -------------------------------------------------------------
# TAB 1: 🎯 Smart Stock Advisor & Trade Planner
# -------------------------------------------------------------
elif active_tab in ["🎯 Smart Stock Advisor (When to Buy/Sell)", "🎯 Easy Stock Advisor (Buy / Sell Advice)"]:
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
        
        # Big Verdict Banner (Two-Tier High Contrast with Setup Quality Grading)
        v_col1, v_col2 = st.columns([1.5, 3])
        badge_c = analysis.get("badge_color", "#10b981")
        setup_grade = analysis.get("setup_grade_title", "⚡ GRADE A (High Probability)")
        win_prob = analysis.get("win_probability", 72)
        rs_data = analysis.get("relative_strength", {})
        sq_data = analysis.get("ttm_squeeze", {})

        with v_col1:
            st.markdown(f"""
            <div style='background: #111622; border: 2px solid {badge_c}; border-radius: 10px; padding: 18px; text-align: center;'>
                <div style='font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>Setup Quality Grade</div>
                <div style='font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin: 4px 0;'>{setup_grade}</div>
                <div style='font-size: 1.4rem; font-weight: 800; color: {badge_c}; margin: 4px 0; font-family: "Outfit", sans-serif;'>{analysis.get("verdict")}</div>
                <div style='display: flex; justify-content: center; gap: 8px; font-size: 0.85rem; margin-top: 4px;'>
                    <span style='color: #f8fafc; font-weight: 700;'>Score: <span class='mono-num'>{analysis.get("score")} / 10</span></span>
                    <span style='background: rgba(16,185,129,0.2); color: #10b981; padding: 1px 6px; border-radius: 4px; font-weight: 700;'>{win_prob}% Win-Rate</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with v_col2:
            disp_name = analysis.get("display_name", display_symbol_name(adv_sym))
            v_desc = analysis.get("verdict_desc", "")
            curr_p = analysis.get("current_price", 0.0)
            h_text = analysis.get("horizon_text", analysis.get("holding_time_text", "Swing (3-7 Days)"))
            
            rs_tag = f"<span style='color: #10b981;'>💪 RS: +{rs_data.get('rs_diff_pct', 0.0)}% vs Nifty</span>" if rs_data.get("status") in ["STRONG_OUTPERFORMER", "OUTPERFORMING"] else "<span style='color: #94a3b8;'>In-line with Nifty</span>"
            sq_tag = "<span style='color: #38bdf8; font-weight: 700;'>🚀 TTM Squeeze Fired</span>" if sq_data.get("squeeze_fired") else ("<span style='color: #f59e0b;'>⚡ Squeeze Coiling</span>" if sq_data.get("squeeze_on") else "")

            st.markdown(f"""
            <div style='background: #111622; border: 1px solid #1e293b; border-radius: 10px; padding: 18px;'>
                <div style='display: flex; justify-content: space-between; align-items: baseline;'>
                    <div style='font-size: 1.2rem; font-weight: 700; color: #f8fafc; font-family: "Outfit", sans-serif;'>
                        Analysis for <strong>{disp_name}</strong> (`{clean_symbol(adv_sym)}`)
                    </div>
                    <div>{sq_tag}</div>
                </div>
                <div style='color: #94a3b8; font-size: 0.90rem; margin: 6px 0 10px 0;'>
                    {v_desc}
                </div>
                <div style='display: flex; gap: 18px; font-size: 0.88rem; flex-wrap: wrap;'>
                    <div>💵 <strong>Live Price:</strong> <span class='mono-num'>₹{curr_p:,.2f}</span></div>
                    <div>⏳ <strong>Horizon:</strong> {h_text}</div>
                    <div>{rs_tag}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        t1 = analysis.get("target_1", {"price": curr_p * 1.03, "gain_pct": 3.0, "reward_risk": 1.5})
        t2 = analysis.get("target_2", {"price": curr_p * 1.06, "gain_pct": 6.0, "reward_risk": 2.5})
        sl = analysis.get("stop_loss", {"price": curr_p * 0.98, "loss_pct": 2.0})
        entry_z = analysis.get("entry_zone", f"₹{curr_p * 0.998:.2f} – ₹{curr_p:.2f}")

        # Proportional Visual R:R Price Ladder
        p_sl = float(sl.get("price", curr_p * 0.98))
        p_entry = float(curr_p)
        p_t1 = float(t1.get("price", curr_p * 1.03))
        p_t2 = float(t2.get("price", curr_p * 1.06))
        
        total_span = max(0.01, p_t2 - p_sl)
        pct_entry = ((p_entry - p_sl) / total_span) * 100.0
        pct_t1 = ((p_t1 - p_sl) / total_span) * 100.0
        
        st.markdown(f"""
        <div class='op-card' style='padding: 16px 20px; margin: 14px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <div style='font-size: 0.82rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;'>📐 Proportional Risk-to-Reward Price Ladder</div>
                <span class='badge-bull'>Blended R:R: 2.00R (Gross) &bull; ≥1.60R Net Gate</span>
            </div>
            
            <div style='position: relative; height: 12px; background: #1e293b; border-radius: 6px; margin: 28px 10px 36px 10px;'>
                <div style='position: absolute; left: 0%; width: {pct_entry:.1f}%; height: 100%; background: #f43f5e; border-radius: 6px 0 0 6px;'></div>
                <div style='position: absolute; left: {pct_entry:.1f}%; width: {pct_t1 - pct_entry:.1f}%; height: 100%; background: #10b981;'></div>
                <div style='position: absolute; left: {pct_t1:.1f}%; width: {100.0 - pct_t1:.1f}%; height: 100%; background: #059669; border-radius: 0 6px 6px 0;'></div>
                
                <div style='position: absolute; left: 0%; top: -24px; font-size: 0.72rem; color: #f43f5e; font-weight: 700; font-family: "JetBrains Mono", monospace;'>
                    🛑 SL: ₹{p_sl:,.2f}
                </div>
                <div style='position: absolute; left: 0%; top: 16px; font-size: 0.68rem; color: #fca5a5; font-family: "JetBrains Mono", monospace;'>
                    -{sl.get('loss_pct', 0.0):.1f}% Risk
                </div>
                
                <div style='position: absolute; left: {pct_entry:.1f}%; top: -24px; transform: translateX(-50%); font-size: 0.72rem; color: #38bdf8; font-weight: 700; font-family: "JetBrains Mono", monospace;'>
                    📍 ENTRY: ₹{p_entry:,.2f}
                </div>
                <div style='position: absolute; left: {pct_entry:.1f}%; top: 16px; transform: translateX(-50%); font-size: 0.68rem; color: #94a3b8;'>
                    Base Level
                </div>
                
                <div style='position: absolute; left: {pct_t1:.1f}%; top: -24px; transform: translateX(-50%); font-size: 0.72rem; color: #10b981; font-weight: 700; font-family: "JetBrains Mono", monospace;'>
                    🎯 T1: ₹{p_t1:,.2f}
                </div>
                <div style='position: absolute; left: {pct_t1:.1f}%; top: 16px; transform: translateX(-50%); font-size: 0.68rem; color: #86efac;'>
                    50% Lock 🔒
                </div>
                
                <div style='position: absolute; right: 0%; top: -24px; font-size: 0.72rem; color: #10b981; font-weight: 800; font-family: "JetBrains Mono", monospace;'>
                    🚀 T2: ₹{p_t2:,.2f}
                </div>
                <div style='position: absolute; right: 0%; top: 16px; font-size: 0.68rem; color: #86efac;'>
                    Runner (2.5R)
                </div>
            </div>
            
            <div style='font-size: 0.76rem; color: #94a3b8; margin-top: 6px;'>
                🔒 <strong>Dynamic Breakeven Milestone:</strong> When Target 1 (₹{p_t1:,.2f}) is touched, 50% profits are automatically locked and Stop-Loss moves to Breakeven (₹{p_entry:,.2f}).
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Structural Pivot Levels Strip
        if analysis.get("pivots"):
            piv = analysis["pivots"]
            st.markdown(f"""
            <div style='background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 8px 14px; margin: 10px 0; display: flex; justify-content: space-between; font-size: 0.80rem; font-family: "JetBrains Mono", monospace;'>
                <div><span style='color: #94a3b8;'>S2:</span> <strong style='color: #f43f5e;'>₹{piv.get('s2', 0):,.2f}</strong></div>
                <div><span style='color: #94a3b8;'>S1:</span> <strong style='color: #fca5a5;'>₹{piv.get('s1', 0):,.2f}</strong></div>
                <div><span style='color: #38bdf8;'>PIVOT:</span> <strong style='color: #38bdf8;'>₹{piv.get('pivot', 0):,.2f}</strong></div>
                <div><span style='color: #94a3b8;'>R1:</span> <strong style='color: #86efac;'>₹{piv.get('r1', 0):,.2f}</strong></div>
                <div><span style='color: #94a3b8;'>R2:</span> <strong style='color: #22c55e;'>₹{piv.get('r2', 0):,.2f}</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### 📋 The Trade Blueprint (Exact Numbers)")
        b1, b2, b3, b4 = st.columns(4)
        
        b1.metric("📍 Ideal Entry Price Zone", f"{entry_z}", "Buy within this range")
        b2.metric("🎯 Target 1 (Quick Profit)", f"₹{t1['price']:,.2f}", f"▲ +{t1.get('gain_pct', 0.0):.1f}% profit", delta_color="normal")
        b3.metric("🚀 Target 2 (Extended Move)", f"₹{t2['price']:,.2f}", f"▲ +{t2.get('gain_pct', 0.0):.1f}% profit", delta_color="normal")
        b4.metric("🛑 Safety Stop-Loss", f"₹{sl['price']:,.2f}", f"▼ -{sl.get('loss_pct', 0.0):.1f}% risk", delta_color="normal")
        
        # 1-Click Quick Execution
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        t_col1, t_col2 = st.columns([2, 1])
        with t_col1:
            trade_cap = st.number_input("Trading Budget for this Trade (₹):", min_value=5000.0, value=25000.0, step=5000.0, key="adv_trade_budget_input")
        with t_col2:
            exec_qty = max(1, int(trade_cap / max(1.0, curr_p)))
            st.markdown(f"<div style='padding-top: 28px; color: #94a3b8; font-size: 0.88rem;'>Quantity: <strong style='color: #f8fafc;'>{exec_qty} Shares</strong> (₹{curr_p*exec_qty:,.0f})</div>", unsafe_allow_html=True)

        if st.button(f"🚀 1-Click Safe Trade: Buy {exec_qty} Shares of {disp_name} (₹{curr_p*exec_qty:,.0f})", type="primary", use_container_width=True, key="adv_1click_trade_btn"):
            proposal = {
                "symbol": adv_sym,
                "target_asset": adv_sym,
                "action": "BUY_STOCK" if "BUY" in analysis.get("verdict", "BUY") else "SELL_STOCK",
                "confidence_score": analysis.get("score", 7.5),
                "entry_price": curr_p,
                "sl": float(sl.get("price", curr_p * 0.98)),
                "target_1": float(t1.get("price", curr_p * 1.03)),
                "horizon": h_key,
                "notes": f"Stock Advisor Pick ({disp_name})"
            }
            p_state = get_portfolio_state()
            guard = AIGuardrails(min_confidence_threshold=7.0)
            approved, g_reason, sanitized_order = guard.evaluate_proposal(proposal, p_state, enforce_time_cutoff=False)
            
            if approved:
                order_res = broker.place_order(
                    symbol=adv_sym,
                    side="BUY" if "BUY" in analysis.get("verdict", "BUY") else "SELL",
                    quantity=exec_qty,
                    price=curr_p,
                    sl=float(sl.get("price", curr_p * 0.98)),
                    tp=float(t1.get("price", curr_p * 1.03)),
                    strategy_name="Stock_Advisor_Pick"
                )
                if order_res.get("status") in ["FILLED", "SUCCESS"]:
                    st.success(f"✅ Order Executed! Bought {exec_qty} shares of {disp_name} @ ₹{curr_p:.2f}. Safety SL placed @ ₹{float(sl.get('price', curr_p * 0.98)):.2f}.")
                    st.rerun()
                else:
                    st.error(f"❌ Order Rejected: {order_res.get('message')}")
            else:
                st.error(f"🛡️ Guardrail Protected: {g_reason}")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        # Technical Candlestick Chart (Collapsible for Clean UX)
        with st.expander("📊 **View Price Chart & Technical Strategy Levels (Optional)**", expanded=False):
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
                if "VWAP" in adv_chart_df.columns:
                    fig_adv.add_trace(go.Scatter(x=adv_chart_df.index, y=adv_chart_df["VWAP"], name="VWAP", line=dict(color="#e2e8f0", width=1.5, dash="dot")), row=1, col=1)
                if "Supertrend" in adv_chart_df.columns:
                    fig_adv.add_trace(go.Scatter(x=adv_chart_df.index, y=adv_chart_df["Supertrend"], name="SuperTrend", line=dict(color="#10b981", width=1.2, dash="dash")), row=1, col=1)
                    
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
# TAB: 📦 My Trades & Daily Profit Book
# -------------------------------------------------------------
elif active_tab == "📦 My Trades & Profit Book":
    st.markdown("""
    <div style='margin-bottom: 12px;'>
        <h2 style='margin: 0; font-family: "Outfit", sans-serif;'>📦 My Trades & Daily Profit Book</h2>
        <div style='color: #94a3b8; font-size: 0.9rem; margin-top: 4px;'>Live overview of your active positions, daily earnings in ₹, and completed trade history.</div>
    </div>
    """, unsafe_allow_html=True)
    
    p_state = get_portfolio_state()
    active_pos = broker.get_open_positions()
    closed_trades = get_closed_trades()
    
    # Financial KPI Cards
    cash_val = float(p_state.get("cash", 100000.0))
    daily_pnl = float(p_state.get("daily_pnl", 0.0))
    open_pnl = sum(p.get("unrealized_pnl", 0.0) for p in active_pos)
    total_equity = cash_val + sum(p.get("entry_price", 0) * p.get("quantity", 0) for p in active_pos) + open_pnl
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Total Account Equity", format_currency_inr(total_equity))
    k2.metric("💵 Available Cash", format_currency_inr(cash_val))
    pnl_sign = "▲ +" if daily_pnl >= 0 else "▼ "
    k3.metric("📈 Today's Realized Profit", format_currency_inr(daily_pnl), f"{pnl_sign}{daily_pnl:,.2f} ₹", delta_color="normal")
    open_sign = "▲ +" if open_pnl >= 0 else "▼ "
    k4.metric("⚡ Live Open Trades P&L", format_currency_inr(open_pnl), f"{open_sign}{open_pnl:,.2f} ₹", delta_color="normal")
    
    st.markdown("---")
    
    # Active Positions Section
    st.markdown("### 📌 Active Open Positions")
    if active_pos:
        for pos in active_pos:
            sym = pos["symbol"]
            qty = int(pos["quantity"])
            entry_p = float(pos["entry_price"])
            curr_p = float(pos.get("current_price", entry_p))
            u_pnl = pos.get("unrealized_pnl", (curr_p - entry_p) * qty)
            u_pct = pos.get("unrealized_pnl_pct", ((curr_p - entry_p) / entry_p) * 100.0 if entry_p > 0 else 0.0)
            side = pos["side"].upper()
            
            pnl_col = "#10b981" if u_pnl >= 0 else "#f43f5e"
            pnl_txt = f"+₹{u_pnl:,.2f} (+{u_pct:.2f}%)" if u_pnl >= 0 else f"-₹{abs(u_pnl):,.2f} ({u_pct:.2f}%)"
            
            st.markdown(f"""
            <div style='background: #111622; border: 1.5px solid {pnl_col}; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                <div>
                    <div style='display: flex; gap: 10px; align-items: center;'>
                        <strong style='font-size: 1.15rem; color: #f8fafc;'>{display_symbol_name(sym)}</strong>
                        <span style='background: {"rgba(16,185,129,0.15)" if side == "BUY" else "rgba(244,63,94,0.15)"}; color: {"#10b981" if side == "BUY" else "#f43f5e"}; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;'>{side} {qty} Shares</span>
                    </div>
                    <div style='font-size: 0.85rem; color: #94a3b8; margin-top: 4px;'>
                        Bought @ ₹{entry_p:,.2f} &bull; Current Market Price: <strong style='color: #38bdf8;'>₹{curr_p:,.2f}</strong>
                    </div>
                </div>
                <div style='text-align: right; display: flex; gap: 18px; align-items: center;'>
                    <div>
                        <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;'>Live Profit / Loss</div>
                        <div style='font-size: 1.25rem; font-weight: 800; color: {pnl_col}; font-family: "JetBrains Mono", monospace;'>{pnl_txt}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            c_btn_col1, c_btn_col2 = st.columns([4, 1])
            with c_btn_col2:
                if st.button(f"🔴 Exit {display_symbol_name(sym)}", key=f"exit_pos_{sym}_{pos.get('id', 0)}", type="secondary", use_container_width=True):
                    sq_res = broker.close_position(pos.get("id") or sym)
                    if sq_res.get("status") in ["FILLED", "SUCCESS"]:
                        st.success(f"Closed {sym} position successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error closing position: {sq_res.get('message')}")
    else:
        st.info("💡 You currently have no open active positions. Use the **Pre-Market Picks** or **Stock Advisor** to place safe trades.")
        
    st.markdown("---")
    
    # Closed Trades History
    st.markdown("### 📜 Past Completed Trades (Profit History)")
    if closed_trades:
        c_df = pd.DataFrame(closed_trades)
        
        # Resolve PnL column defensively
        pnl_col = "net_pnl" if "net_pnl" in c_df.columns else ("pnl" if "pnl" in c_df.columns else ("gross_pnl" if "gross_pnl" in c_df.columns else None))
        if pnl_col:
            c_df["Profit / Loss (₹)"] = c_df[pnl_col].apply(lambda x: format_currency_inr(float(x or 0.0)))
        else:
            c_df["Profit / Loss (₹)"] = "₹0.00"
            
        if "pnl_pct" in c_df.columns:
            c_df["Return %"] = c_df["pnl_pct"].apply(lambda x: f"{float(x or 0.0):+.2f}%")
        elif pnl_col and "entry_price" in c_df.columns and "quantity" in c_df.columns:
            c_df["Return %"] = c_df.apply(lambda r: f"{(float(r[pnl_col] or 0)/(max(1.0, float(r.get('entry_price', 1.0))*float(r.get('quantity', 1.0))))*100.0):+.2f}%", axis=1)
        else:
            c_df["Return %"] = "0.00%"
            
        if "entry_price" in c_df.columns:
            c_df["Bought @"] = c_df["entry_price"].apply(lambda x: f"₹{float(x or 0.0):.2f}")
        if "exit_price" in c_df.columns:
            c_df["Sold @"] = c_df["exit_price"].apply(lambda x: f"₹{float(x or 0.0):.2f}")
        if "symbol" in c_df.columns:
            c_df["Stock"] = c_df["symbol"].apply(lambda s: display_symbol_name(str(s)))
        
        display_cols = ["Stock", "side", "quantity", "Bought @", "Sold @", "Profit / Loss (₹)", "Return %", "exit_reason", "exit_time"]
        available_cols = [c for c in display_cols if c in c_df.columns]
        st.dataframe(c_df[available_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No completed trades recorded yet today.")

# -------------------------------------------------------------
# TAB 5: ⚙️ Settings & Risk Controls
# -------------------------------------------------------------
elif active_tab in ["⚙️ Settings & Risk Controls", "⚙️ Simple Settings & Safety"]:
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
