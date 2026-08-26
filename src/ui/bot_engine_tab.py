"""
Automated Live / Paper Bot Engine Tab.
"""

import streamlit as st
from src.strategies import AVAILABLE_STRATEGIES, get_strategy
from src.brokers import get_broker
from src.utils.helpers import format_currency_inr, display_symbol_name, format_holding_duration

def render_bot_engine_tab(broker_instance):
    """Renders the Automated Live / Paper Bot Controls and Real-Time Positions."""
    st.markdown("""
    <h2>⚡ Automated Bot Engine</h2>
    <div style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 14px;'>Let the bot automatically scan Indian stocks, take trades according to your rules, and protect your capital.</div>
    """, unsafe_allow_html=True)
    
    bot = st.session_state.bot_instance
    
    b_col1, b_col2, b_col3, b_col4 = st.columns([2, 2, 2, 2])
    with b_col3:
        bot_strat = st.selectbox("Strategy to Use:", list(AVAILABLE_STRATEGIES.keys()), index=0)
        
    with b_col4:
        bot_tf = st.selectbox("Candle Speed:", ["1m", "5m", "15m", "30m"], index=1)
        
    bot.strategy_name = bot_strat
    bot.strategy = get_strategy(bot_strat)
    bot.timeframe = bot_tf
    
    interval_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800}
    int_sec = interval_map.get(bot_tf, 60)

    with b_col1:
        if bot.is_running:
            if st.button("⏹️ Pause Bot", type="secondary", use_container_width=True):
                bot.stop_continuous()
                st.session_state.bot_running = False
                st.rerun()
        else:
            if st.button("▶️ Turn Bot ON", type="primary", use_container_width=True):
                bot.start_continuous(interval_sec=int_sec)
                st.session_state.bot_running = True
                st.rerun()
                
    with b_col2:
        scan_now = st.button("🔄 Scan Market Now", use_container_width=True)

    if bot.is_running:
        st.success(f"🟢 **Bot Engine Active** — Continuously scanning Indian stocks every {bot_tf} ({int_sec}s) and managing trailing stops in background. (Last scan: {bot.last_scan_time or 'Just started...'})")
    else:
        st.info("⚪ **Bot Engine Idle** — Click **'▶️ Turn Bot ON'** to launch autonomous background scanning, or click **'🔄 Scan Market Now'** for on-demand execution.")
    
    if scan_now:
        with st.spinner("Scanning Indian stocks and evaluating trade setups..."):
            scan_res = bot.scan_and_execute()
            if scan_res.get("status") == "SUCCESS":
                st.success(f"Scan complete at {scan_res['last_scan']}. Checked {scan_res.get('signals_count', 0)} signal(s) and managed open trades.")
            elif scan_res.get("status") == "HALTED":
                st.error(scan_res.get("message"))
                
    st.markdown("---")
    
    @st.fragment(run_every=2)
    def render_active_open_positions():
        broker_local = get_broker(st.session_state.active_broker_name)
        active_pos = broker_local.get_open_positions()
        
        st.markdown("### 📌 Active Open Positions (Live 2s Daemon Stream)")
        
        if active_pos:
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
                    
                    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 14px;'>
                        <div style='background: #080b11; border: 1px solid #1e293b; border-radius: 8px; padding: 10px;'>
                            <div style='color: #94a3b8; font-size: 0.72rem; text-transform: uppercase;'>Quantity</div>
                            <div style='color: #f8fafc; font-size: 1.15rem; font-weight: 800; font-family: "JetBrains Mono", monospace;'>{qty} Shares</div>
                        </div>
                        <div style='background: #080b11; border: 1px solid #1e293b; border-radius: 8px; padding: 10px;'>
                            <div style='color: #94a3b8; font-size: 0.72rem; text-transform: uppercase;'>Bought @</div>
                            <div style='color: #f8fafc; font-size: 1.15rem; font-weight: 800; font-family: "JetBrains Mono", monospace;'>₹{entry_p:,.2f}</div>
                        </div>
                        <div style='background: #080b11; border: 1px solid #1e293b; border-radius: 8px; padding: 10px;'>
                            <div style='color: #94a3b8; font-size: 0.72rem; text-transform: uppercase;'>Live Market Price</div>
                            <div style='color: #38bdf8; font-size: 1.15rem; font-weight: 800; font-family: "JetBrains Mono", monospace;'>₹{curr_p:,.2f}</div>
                        </div>
                        <div style='background: #080b11; border: 1px solid #1e293b; border-radius: 8px; padding: 10px;'>
                            <div style='color: #94a3b8; font-size: 0.72rem; text-transform: uppercase;'>Total Exposure</div>
                            <div style='color: #f8fafc; font-size: 1.15rem; font-weight: 800; font-family: "JetBrains Mono", monospace;'>₹{exposure:,.2f}</div>
                        </div>
                        <div style='background: #080b11; border: 1px solid {card_border}; border-radius: 8px; padding: 10px;'>
                            <div style='color: #94a3b8; font-size: 0.72rem; text-transform: uppercase;'>Live P&L</div>
                            <div style='color: {card_border}; font-size: 1.15rem; font-weight: 800; font-family: "JetBrains Mono", monospace;'>{format_currency_inr(pnl)} ({pnl_arr}{pnl_p:+.2f}%)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                pos_btn_c1, pos_btn_c2 = st.columns([5, 1])
                with pos_btn_c2:
                    if st.button(f"🔴 Exit Leg", key=f"btn_close_leg_{pos['symbol']}_{pos.get('id', 0)}", type="secondary", use_container_width=True):
                        close_res = broker_local.close_position(pos.get("id") or pos["symbol"])
                        st.success(f"Closed {pos['symbol']} successfully!")
                        st.rerun()
        else:
            st.info("ℹ️ No active bot positions currently open.")
            
    render_active_open_positions()
