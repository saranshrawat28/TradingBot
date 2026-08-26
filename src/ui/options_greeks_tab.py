"""
Options Greeks, Max Pain, PCR & Symmetrical Option Chain Matrix Tab.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data.data_fetcher import get_live_quote
from src.strategies.options_greeks import OptionChainBuilder, SmartStrikeSelector
from src.engine.ai_guardrails import AIGuardrails
from src.utils.storage import get_portfolio_state

def render_options_greeks_tab(broker_instance):
    """Renders the European Black-Scholes Greeks Engine & Interactive Option Chain."""
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
            sl_p = round(entry_p * 0.80, 2)
            t1_p = round(entry_p * 1.30, 2)
            t2_p = round(entry_p * 1.60, 2)
            
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
            
            p_state = get_portfolio_state()
            guard = AIGuardrails(min_confidence_threshold=7.5)
            approved, reason, sanitized_order = guard.evaluate_proposal(proposal, p_state, enforce_time_cutoff=False)
            
            if approved:
                order_res = broker_instance.place_order(
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
