"""
ApexTrade - Indian Stocks & F&O Algorithmic Trading Terminal & Web Dashboard
Institutional Two-Tier Terminal UI Architecture (Modular Component Dispatcher)
"""

import streamlit as st
import config
from src.brokers import get_broker
from src.engine.live_bot import LiveTradingBot
from src.ui import (
    inject_terminal_css,
    render_live_header,
    render_live_stock_watcher,
    render_sidebar_navigation,
    render_pre_market_tab,
    render_chat_assistant_tab,
    render_autonomous_tab,
    render_options_greeks_tab,
    render_stock_advisor_tab,
    render_backtester_tab,
    render_bot_engine_tab,
    render_screener_tab,
    render_portfolio_tab,
    render_settings_tab,
    render_quant_research_tab
)

# -------------------------------------------------------------
# 1. Streamlit Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="ApexTrade Terminal | Indian Stock & F&O AI Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 2. Session State Initialization
# -------------------------------------------------------------
if "active_broker_name" not in st.session_state:
    st.session_state.active_broker_name = config.ACTIVE_BROKER

if "bot_instance" not in st.session_state:
    st.session_state.bot_instance = LiveTradingBot(broker_name=st.session_state.active_broker_name)

if "bot_running" not in st.session_state:
    st.session_state.bot_running = False

# Active broker instance
broker = get_broker(st.session_state.active_broker_name)

# -------------------------------------------------------------
# 3. Global CSS Styling & Top Navigation Telemetry
# -------------------------------------------------------------
inject_terminal_css()
render_live_header(broker)
render_live_stock_watcher()

# -------------------------------------------------------------
# 4. Sidebar Controls & Tab Selection
# -------------------------------------------------------------
ui_mode, active_tab = render_sidebar_navigation(broker)
is_simple = "Simple" in ui_mode

# -------------------------------------------------------------
# 5. Modular Tab Routing
# -------------------------------------------------------------
if active_tab == "🗣️ Talk to Your AI Bot (Chat & Voice)":
    render_chat_assistant_tab(broker)

elif active_tab == "🌅 Pre-Market & Best Stocks Today":
    render_pre_market_tab(broker)

elif active_tab in ["🤖 Autonomous AI Trading Agent (Claude / Kimi / F&O)", "🤖 AI Auto-Pilot (Automated Safe Trading)"]:
    render_autonomous_tab(broker)

elif active_tab == "🔬 Systematic Quant Research Lab":
    render_quant_research_tab(broker)

elif active_tab == "⚡ NFO Options Greeks & OI Matrix":
    render_options_greeks_tab(broker)

elif active_tab in ["🎯 Smart Stock Advisor (When to Buy/Sell)", "🎯 Easy Stock Advisor (Buy / Sell Advice)"]:
    render_stock_advisor_tab(broker, is_simple_mode=is_simple)

elif active_tab == "📊 Strategy Backtester (Test Any Stock)":
    render_backtester_tab(broker)

elif active_tab == "⚡ Automated Live / Paper Bot":
    render_bot_engine_tab(broker)

elif active_tab == "🔍 Indian Market Screener (Scan All Stocks)":
    render_screener_tab(broker)

elif active_tab == "📦 My Trades & Profit Book":
    render_portfolio_tab(broker)

elif active_tab in ["⚙️ Settings & Risk Controls", "⚙️ Simple Settings & Safety"]:
    render_settings_tab(broker, is_simple_mode=is_simple)

else:
    render_pre_market_tab(broker)
