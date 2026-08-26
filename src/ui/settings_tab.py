"""
Settings, Risk Controls & Broker Connections Tab.
"""

import streamlit as st
import config
from src.engine.live_bot import LiveTradingBot
from src.utils.storage import reset_all_data

def render_settings_tab(broker_instance, is_simple_mode: bool = False):
    """Renders the Settings, Risk Parameters, Broker & LLM Key management."""
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
