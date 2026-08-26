"""
Real-Time Multi-Sector Indian Market Screener Tab.
Persists screener table in Streamlit session state for instant tab switching.
"""

import streamlit as st
import pandas as pd
import config
from src.data.data_fetcher import get_historical_data
from src.strategies.indicators import add_all_indicators

def render_screener_tab(broker_instance):
    """Renders the Real-Time Market Screener across Indian Sectors."""
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
                
            st.session_state["screener_results"] = screener_rows

    if "screener_results" in st.session_state:
        screener_rows = st.session_state["screener_results"]
        if screener_rows:
            res_df = pd.DataFrame(screener_rows)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No data returned for selected sector.")
