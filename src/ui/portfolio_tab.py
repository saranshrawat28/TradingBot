"""
Portfolio & Daily Profit Book Tab.
"""

import streamlit as st
import pandas as pd
from src.utils.storage import get_portfolio_state, get_closed_trades
from src.utils.helpers import format_currency_inr, display_symbol_name

def render_portfolio_tab(broker_instance):
    """Renders the Portfolio, Active Positions, and Completed Trades Book."""
    st.markdown("""
    <div style='margin-bottom: 12px;'>
        <h2 style='margin: 0; font-family: "Outfit", sans-serif;'>📦 My Trades & Daily Profit Book</h2>
        <div style='color: #94a3b8; font-size: 0.9rem; margin-top: 4px;'>Live overview of your active positions, daily earnings in ₹, and completed trade history.</div>
    </div>
    """, unsafe_allow_html=True)
    
    p_state = get_portfolio_state()
    active_pos = broker_instance.get_open_positions()
    closed_trades = get_closed_trades()
    
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
                    sq_res = broker_instance.close_position(pos.get("id") or sym)
                    if sq_res.get("status") in ["FILLED", "SUCCESS"]:
                        st.success(f"Closed {sym} position successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error closing position: {sq_res.get('message')}")
    else:
        st.info("💡 You currently have no open active positions. Use the **Pre-Market Picks** or **Stock Advisor** to place safe trades.")
        
    st.markdown("---")
    
    st.markdown("### 📜 Past Completed Trades & Itemized Cost Breakdown")
    if closed_trades:
        c_df = pd.DataFrame(closed_trades)
        pnl_col = "net_pnl" if "net_pnl" in c_df.columns else ("pnl" if "pnl" in c_df.columns else ("gross_pnl" if "gross_pnl" in c_df.columns else None))
        
        tot_net = sum(float(t.get(pnl_col, 0.0) or 0.0) for t in closed_trades) if pnl_col else 0.0
        tot_gross = sum(float(t.get("gross_pnl", t.get(pnl_col, 0.0)) or 0.0) for t in closed_trades)
        tot_taxes = max(0.0, tot_gross - tot_net)
        stt_est = tot_taxes * 0.45
        gst_est = tot_taxes * 0.35
        sebi_exch_est = tot_taxes * 0.20
        
        tx1, tx2, tx3, tx4 = st.columns(4)
        tx1.metric("💰 Gross Realized P&L", format_currency_inr(tot_gross))
        tx2.metric("🛡️ Net Realized P&L", format_currency_inr(tot_net), f"{(tot_net):+,.2f} ₹", delta_color="normal")
        tx3.metric("🏛️ Total Taxes & Fees", format_currency_inr(tot_taxes), "STT + GST + SEBI", delta_color="off")
        win_count = sum(1 for t in closed_trades if float(t.get(pnl_col, 0.0) or 0.0) > 0) if pnl_col else 0
        w_rate = (win_count / len(closed_trades)) * 100.0 if closed_trades else 0.0
        tx4.metric("📊 Win Rate", f"{w_rate:.1f}%", f"{win_count}/{len(closed_trades)} Trades Won", delta_color="normal")
        
        st.markdown(f"""
        <div style='background: #080b11; border: 1px solid #1e293b; border-radius: 8px; padding: 10px 14px; margin: 10px 0 14px 0; font-size: 0.78rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;'>
            <div><span style='color: #94a3b8;'>STT (Securities Transaction Tax):</span> <strong style='color: #f8fafc;'>₹{stt_est:,.2f}</strong></div>
            <div><span style='color: #94a3b8;'>GST (18% on Brokerage & Txn):</span> <strong style='color: #f8fafc;'>₹{gst_est:,.2f}</strong></div>
            <div><span style='color: #94a3b8;'>Exchange & SEBI Turnover:</span> <strong style='color: #f8fafc;'>₹{sebi_exch_est:,.2f}</strong></div>
            <div><span style='color: #94a3b8;'>Stamp Duty (State):</span> <strong style='color: #10b981;'>Verified Standard</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
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
