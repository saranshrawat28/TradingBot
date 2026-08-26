"""
Modular Streamlit UI Tabs & Components for ApexTrade Terminal.
"""

from src.ui.components import inject_terminal_css, render_live_header, render_live_stock_watcher, render_sidebar_navigation
from src.ui.pre_market_tab import render_pre_market_tab
from src.ui.ai_chat_tab import render_chat_assistant_tab
from src.ui.autonomous_tab import render_autonomous_tab
from src.ui.options_greeks_tab import render_options_greeks_tab
from src.ui.stock_advisor_tab import render_stock_advisor_tab
from src.ui.backtester_tab import render_backtester_tab
from src.ui.bot_engine_tab import render_bot_engine_tab
from src.ui.screener_tab import render_screener_tab
from src.ui.portfolio_tab import render_portfolio_tab
from src.ui.settings_tab import render_settings_tab
from src.ui.research_tab import render_quant_research_tab

__all__ = [
    "inject_terminal_css",
    "render_live_header",
    "render_live_stock_watcher",
    "render_sidebar_navigation",
    "render_pre_market_tab",
    "render_chat_assistant_tab",
    "render_autonomous_tab",
    "render_options_greeks_tab",
    "render_stock_advisor_tab",
    "render_backtester_tab",
    "render_bot_engine_tab",
    "render_screener_tab",
    "render_portfolio_tab",
    "render_settings_tab",
    "render_quant_research_tab"
]
