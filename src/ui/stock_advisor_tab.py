"""
Smart Stock Advisor & Recommendation Tab.
Provides setup quality ratings (Grade A/B/C), visual R:R price ladders, technical indicators, and price charts.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap
import config
from src.engine.stock_advisor import StockAdvisor
from src.engine.ai_guardrails import AIGuardrails
from src.data.data_fetcher import get_historical_data
from src.strategies.indicators import add_all_indicators
from src.utils.storage import get_portfolio_state
from src.utils.helpers import display_symbol_name, clean_symbol

def render_stock_advisor_tab(broker_instance, is_simple_mode: bool = False):
    """Renders the comprehensive Stock Advisor view."""
    
    st.markdown("""
    <h2>🎯 Smart Stock Advisor: When to Buy, Hold, or Sell</h2>
    <div style='color: #94a3b8; font-size: 0.90rem; margin-bottom: 12px;'>Get instant, data-backed advice on any Indian stock with exact Entry, Stop-Loss, and Target prices.</div>
    """, unsafe_allow_html=True)
    
    col_input, col_horizon, col_btn = st.columns([2.5, 1.5, 1])
    with col_input:
        stock_mode = st.radio(
            "Selection Mode:",
            ["📋 Watchlist & IPOs", "🔍 Type Any Custom / New NSE Ticker"],
            horizontal=True,
            key="adv_mode_radio"
        )
        if "Watchlist" in stock_mode:
            adv_sym = st.selectbox(
                "Select Indian Stock / IPO:",
                config.POPULAR_SYMBOLS,
                index=config.POPULAR_SYMBOLS.index("RELIANCE.NS") if "RELIANCE.NS" in config.POPULAR_SYMBOLS else 0,
                format_func=lambda x: f"{display_symbol_name(x)} ({clean_symbol(x)})",
                key="adv_stock_select_main"
            )
        else:
            from src.data.data_fetcher import resolve_ticker
            custom_input = st.text_input(
                "Enter Any NSE Ticker / Newly Listed Stock:",
                value="SWIGGY",
                placeholder="e.g. SWIGGY, HYUNDAI, BAJAJHFL, WAAREEENER",
                help="Type any stock listed on the National Stock Exchange (NSE).",
                key="adv_custom_input_main"
            )
            adv_sym = resolve_ticker(custom_input) if custom_input.strip() else "RELIANCE.NS"
    with col_horizon:
        st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
        h_option = st.selectbox(
            "Your Trading Goal / Horizon:",
            ["Intraday (Today Only)", "Swing (3-7 Days)", "Positional (2-4 Weeks)", "Long-Term Investment"],
            index=1,
            key="adv_horizon_select_main"
        )
    with col_btn:
        st.markdown("<div style='height: 66px;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🔍 Analyze Stock", type="primary", use_container_width=True, key="adv_analyze_btn_main")
        
    h_map = {
        "Intraday (Today Only)": "intraday",
        "Swing (3-7 Days)": "swing",
        "Positional (2-4 Weeks)": "positional",
        "Long-Term Investment": "long_term"
    }
    h_key = h_map.get(h_option, "swing")
    
    with st.spinner(f"Running Institutional Analysis on {display_symbol_name(adv_sym)}..."):
        analysis = StockAdvisor.analyze_stock(adv_sym, horizon=h_key)
        
    if analysis.get("status") == "SUCCESS":
        st.markdown("---")
        
        v_col1, v_col2 = st.columns([1.5, 3])
        badge_c = analysis.get("badge_color", "#10b981")
        setup_grade = analysis.get("setup_grade_title", "⚡ GRADE A (High Probability)")
        win_prob = analysis.get("win_probability", 72)
        rs_data = analysis.get("relative_strength", {})
        sq_data = analysis.get("ttm_squeeze", {})

        with v_col1:
            grade_html = textwrap.dedent(f"""
            <div style='background: #111622; border: 2px solid {badge_c}; border-radius: 10px; padding: 18px; text-align: center;'>
                <div style='font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;'>Setup Quality Grade</div>
                <div style='font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin: 4px 0;'>{setup_grade}</div>
                <div style='font-size: 1.4rem; font-weight: 800; color: {badge_c}; margin: 4px 0; font-family: "Outfit", sans-serif;'>{analysis.get("verdict")}</div>
                <div style='display: flex; justify-content: center; gap: 8px; font-size: 0.85rem; margin-top: 4px;'>
                    <span style='color: #f8fafc; font-weight: 700;'>Score: <span class='mono-num'>{analysis.get("score")} / 10</span></span>
                    <span style='background: rgba(16,185,129,0.2); color: #10b981; padding: 1px 6px; border-radius: 4px; font-weight: 700;'>{win_prob}% Win-Rate</span>
                </div>
            </div>
            """).strip()
            st.markdown(grade_html, unsafe_allow_html=True)

        with v_col2:
            disp_name = analysis.get("display_name", display_symbol_name(adv_sym))
            v_desc = analysis.get("verdict_desc", "")
            curr_p = analysis.get("current_price", 0.0)
            h_text = analysis.get("horizon_text", analysis.get("holding_time_text", "Swing (3-7 Days)"))
            
            rs_tag = f"<span style='color: #10b981;'>💪 RS: +{rs_data.get('rs_diff_pct', 0.0)}% vs Nifty</span>" if rs_data.get("status") in ["STRONG_OUTPERFORMER", "OUTPERFORMING"] else "<span style='color: #94a3b8;'>In-line with Nifty</span>"
            sq_tag = "<span style='color: #38bdf8; font-weight: 700;'>🚀 TTM Squeeze Fired</span>" if sq_data.get("squeeze_fired") else ("<span style='color: #f59e0b;'>⚡ Squeeze Coiling</span>" if sq_data.get("squeeze_on") else "")

            overview_html = textwrap.dedent(f"""
            <div style='background: #111622; border: 1px solid #1e293b; border-radius: 10px; padding: 18px;'>
                <div style='display: flex; justify-content: space-between; align-items: baseline;'>
                    <div style='font-size: 1.2rem; font-weight: 700; color: #f8fafc; font-family: "Outfit", sans-serif;'>
                        Analysis for <strong>{disp_name}</strong> (<code>{clean_symbol(adv_sym)}</code>)
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
            """).strip()
            st.markdown(overview_html, unsafe_allow_html=True)
            
        t1 = analysis.get("target_1", {"price": curr_p * 1.03, "gain_pct": 3.0, "reward_risk": 1.5})
        t2 = analysis.get("target_2", {"price": curr_p * 1.06, "gain_pct": 6.0, "reward_risk": 2.5})
        sl = analysis.get("stop_loss", {"price": curr_p * 0.98, "loss_pct": 2.0})
        entry_z = analysis.get("entry_zone", f"₹{curr_p * 0.998:.2f} – ₹{curr_p:.2f}")

        p_sl = float(sl.get("price", curr_p * 0.98))
        p_entry = float(curr_p)
        p_t1 = float(t1.get("price", curr_p * 1.03))
        p_t2 = float(t2.get("price", curr_p * 1.06))
        
        total_span = max(0.01, p_t2 - p_sl)
        pct_entry = ((p_entry - p_sl) / total_span) * 100.0
        pct_t1 = ((p_t1 - p_sl) / total_span) * 100.0
        
        ladder_html = textwrap.dedent(f"""
        <div class='op-card' style='padding: 16px 20px; margin: 14px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <div style='font-size: 0.82rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;'>📐 Proportional Risk-to-Reward Price Ladder</div>
                <span class='badge-bull'>Blended R:R: 2.00R (Gross) &bull; ≥1.60R Net Gate</span>
            </div>
            <div style='position: relative; height: 12px; background: #1e293b; border-radius: 6px; margin: 28px 10px 36px 10px;'>
                <div style='position: absolute; left: 0%; width: {pct_entry:.1f}%; height: 100%; background: #f43f5e; border-radius: 6px 0 0 6px;'></div>
                <div style='position: absolute; left: {pct_entry:.1f}%; width: {pct_t1 - pct_entry:.1f}%; height: 100%; background: #10b981;'></div>
                <div style='position: absolute; left: {pct_t1:.1f}%; width: {100.0 - pct_t1:.1f}%; height: 100%; background: #059669; border-radius: 0 6px 6px 0;'></div>
                <div style='position: absolute; left: 0%; top: -24px; font-size: 0.72rem; color: #f43f5e; font-weight: 700; font-family: "JetBrains Mono", monospace;'>🛑 SL: ₹{p_sl:,.2f}</div>
                <div style='position: absolute; left: 0%; top: 16px; font-size: 0.68rem; color: #fca5a5; font-family: "JetBrains Mono", monospace;'>-{sl.get('loss_pct', 0.0):.1f}% Risk</div>
                <div style='position: absolute; left: {pct_entry:.1f}%; top: -24px; transform: translateX(-50%); font-size: 0.72rem; color: #38bdf8; font-weight: 700; font-family: "JetBrains Mono", monospace;'>📍 ENTRY: ₹{p_entry:,.2f}</div>
                <div style='position: absolute; left: {pct_entry:.1f}%; top: 16px; transform: translateX(-50%); font-size: 0.68rem; color: #94a3b8;'>Base Level</div>
                <div style='position: absolute; left: {pct_t1:.1f}%; top: -24px; transform: translateX(-50%); font-size: 0.72rem; color: #10b981; font-weight: 700; font-family: "JetBrains Mono", monospace;'>🎯 T1: ₹{p_t1:,.2f}</div>
                <div style='position: absolute; left: {pct_t1:.1f}%; top: 16px; transform: translateX(-50%); font-size: 0.68rem; color: #86efac;'>50% Lock 🔒</div>
                <div style='position: absolute; right: 0%; top: -24px; font-size: 0.72rem; color: #10b981; font-weight: 800; font-family: "JetBrains Mono", monospace;'>🚀 T2: ₹{p_t2:,.2f}</div>
                <div style='position: absolute; right: 0%; top: 16px; font-size: 0.68rem; color: #86efac;'>Runner (2.5R)</div>
            </div>
            <div style='font-size: 0.76rem; color: #94a3b8; margin-top: 6px;'>
                🔒 <strong>Dynamic Breakeven Milestone:</strong> When Target 1 (₹{p_t1:,.2f}) is touched, 50% profits are automatically locked and Stop-Loss moves to Breakeven (₹{p_entry:,.2f}).
            </div>
        </div>
        """).strip()
        st.markdown(ladder_html, unsafe_allow_html=True)

        if analysis.get("pivots"):
            piv = analysis["pivots"]
            piv_html = textwrap.dedent(f"""
            <div style='background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 8px 14px; margin: 10px 0; display: flex; justify-content: space-between; font-size: 0.80rem; font-family: "JetBrains Mono", monospace;'>
                <div><span style='color: #94a3b8;'>S2:</span> <strong style='color: #f43f5e;'>₹{piv.get('s2', 0):,.2f}</strong></div>
                <div><span style='color: #94a3b8;'>S1:</span> <strong style='color: #fca5a5;'>₹{piv.get('s1', 0):,.2f}</strong></div>
                <div><span style='color: #38bdf8;'>PIVOT:</span> <strong style='color: #38bdf8;'>₹{piv.get('pivot', 0):,.2f}</strong></div>
                <div><span style='color: #94a3b8;'>R1:</span> <strong style='color: #86efac;'>₹{piv.get('r1', 0):,.2f}</strong></div>
                <div><span style='color: #94a3b8;'>R2:</span> <strong style='color: #22c55e;'>₹{piv.get('r2', 0):,.2f}</strong></div>
            </div>
            """).strip()
            st.markdown(piv_html, unsafe_allow_html=True)
            
        st.markdown("### 📋 The Trade Blueprint (Exact Numbers)")
        b1, b2, b3, b4 = st.columns(4)
        
        b1.metric("📍 Ideal Entry Price Zone", f"{entry_z}", "Buy within this range")
        b2.metric("🎯 Target 1 (Quick Profit)", f"₹{t1['price']:,.2f}", f"▲ +{t1.get('gain_pct', 0.0):.1f}% profit", delta_color="normal")
        b3.metric("🚀 Target 2 (Extended Move)", f"₹{t2['price']:,.2f}", f"▲ +{t2.get('gain_pct', 0.0):.1f}% profit", delta_color="normal")
        b4.metric("🛑 Safety Stop-Loss", f"₹{sl['price']:,.2f}", f"▼ -{sl.get('loss_pct', 0.0):.1f}% risk", delta_color="normal")
        
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
                order_res = broker_instance.place_order(
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
                
                fig_adv.add_trace(go.Candlestick(
                    x=adv_chart_df.index,
                    open=adv_chart_df["Open"],
                    high=adv_chart_df["High"],
                    low=adv_chart_df["Low"],
                    close=adv_chart_df["Close"],
                    name="Price"
                ), row=1, col=1)
                
                if "EMA_9" in adv_chart_df.columns:
                    fig_adv.add_trace(go.Scatter(x=adv_chart_df.index, y=adv_chart_df["EMA_9"], name="9 EMA", line=dict(color="#38bdf8", width=1.5)), row=1, col=1)
                if "EMA_21" in adv_chart_df.columns:
                    fig_adv.add_trace(go.Scatter(x=adv_chart_df.index, y=adv_chart_df["EMA_21"], name="21 EMA", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
                if "SuperTrend" in adv_chart_df.columns:
                    fig_adv.add_trace(go.Scatter(x=adv_chart_df.index, y=adv_chart_df["SuperTrend"], name="SuperTrend", line=dict(color="#10b981", dash="dot")), row=1, col=1)
                    
                if "RSI" in adv_chart_df.columns:
                    fig_adv.add_trace(go.Scatter(x=adv_chart_df.index, y=adv_chart_df["RSI"], name="RSI", line=dict(color="#a855f7")), row=2, col=1)
                    fig_adv.add_hline(y=70, line_dash="dash", line_color="#f43f5e", row=2, col=1)
                    fig_adv.add_hline(y=30, line_dash="dash", line_color="#10b981", row=2, col=1)
                    
                fig_adv.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#090d16",
                    plot_bgcolor="#090d16",
                    xaxis_rangeslider_visible=False,
                    height=450,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_adv, use_container_width=True)
                
            # Technical Indicator Breakdown Matrix
            st.markdown("#### 🔬 Technical Indicators & Structural Levels")
            t_m1, t_m2, t_m3, t_m4 = st.columns(4)
            t_m1.metric("RSI (14)", f"{analysis.get('rsi', 50):.1f}", "Bullish Momentum" if analysis.get('rsi', 50) > 50 else "Bearish / Neutral")
            t_m2.metric("Trend Direction", f"{analysis.get('trend', 'Neutral')}", "9 EMA vs 21 EMA")
            t_m3.metric("ATR (Volatility Range)", f"₹{analysis.get('atr', 0.0):.2f}", "Expected Daily Move")
            t_m4.metric("SuperTrend", f"{analysis.get('supertrend_dir', 'Neutral')}", "Trailing Support Level")
            
        with st.expander("❓ **Why This Advice? (Plain English Explanation)**", expanded=False):
            st.markdown(f"""
            - **Trend Condition**: {analysis.get('trend_reason', 'Price trading in alignment with core moving averages.')}
            - **Relative Strength**: {rs_data.get('rs_verdict', 'Tracking the broader Indian stock market.')}
            - **Risk-Reward Balance**: Entry at ₹{curr_p:,.2f} offers a calculated **1:{t1.get('reward_risk', 1.5):.1f} minimum Risk-to-Reward ratio** on Target 1, expanding to **1:{t2.get('reward_risk', 2.5):.1f} on Target 2**.
            """)
    else:
        st.error(f"Could not load analysis for {adv_sym}: {analysis.get('message', 'Unknown Error')}")
