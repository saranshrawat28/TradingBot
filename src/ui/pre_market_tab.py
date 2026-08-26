"""
Pre-Market & Daily Best Stock & F&O Opportunity Tab.
Provides morning market sentiment, breakout stock calls, NIFTY/BANKNIFTY option calls, and swing picks.
Persists report in Streamlit session state so switching tabs is instant (0-latency).
"""

import streamlit as st
import pandas as pd
import textwrap
from datetime import datetime
from src.engine.pre_market_analyzer import PreMarketAnalyzer
from src.engine.ai_guardrails import AIGuardrails
from src.utils.storage import get_portfolio_state
from src.utils.helpers import get_ist_now

def render_pre_market_tab(broker_instance):
    """Renders the comprehensive Pre-Market & Best Stocks Today view."""
    
    # 1. Header with Title & Manual Refresh Controls
    head_col1, head_col2 = st.columns([3.5, 1.5])
    with head_col1:
        st.markdown("""
        <div>
            <h2 style='margin: 0; font-size: 1.55rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>🌅 Pre-Market & Best Stock / F&O Calls Today</h2>
            <div style='color: #94a3b8; font-size: 0.88rem; margin-top: 2px;'>Scans 40+ liquid Indian stocks before 9:15 AM to identify opening gaps, high-momentum breakout setups, and weekly option contracts.</div>
        </div>
        """, unsafe_allow_html=True)
    with head_col2:
        last_updated_str = st.session_state.get("pre_market_last_updated", "Just now")
        st.markdown(f"<div style='text-align: right; color: #64748b; font-size: 0.78rem; margin-bottom: 4px;'>🕒 Last scan: {last_updated_str}</div>", unsafe_allow_html=True)
        force_refresh = st.button("🔄 Refresh Pre-Market Scan", type="secondary", use_container_width=True, key="btn_refresh_premarket_data")

    # 2. Check Session State Cache — Only Scan If Not Cached or Refreshed
    if "pre_market_report" not in st.session_state or force_refresh:
        with st.spinner("Analyzing Pre-Market Opening Cues & Scanning 40+ Indian Equities..."):
            report = PreMarketAnalyzer.get_pre_market_report()
            st.session_state["pre_market_report"] = report
            st.session_state["pre_market_last_updated"] = get_ist_now().strftime("%I:%M:%S %p IST")
    else:
        report = st.session_state["pre_market_report"]
        
    sentiment_info = report.get("market_sentiment", {})
    top_picks = report.get("top_picks", [])
    option_calls = report.get("option_calls", [])
    swing_picks = report.get("swing_picks", [])
    gap_ups = report.get("gap_ups", [])
    gap_downs = report.get("gap_downs", [])
    
    # 3. Market Sentiment Summary Banner
    st_bg = sentiment_info.get("badge_bg", "rgba(16, 185, 129, 0.15)")
    st_border = sentiment_info.get("badge_color", "#10b981")
    st_text_color = sentiment_info.get("badge_color", "#10b981")
    
    sentiment_html = textwrap.dedent(f"""
    <div style='background: #0d121f; border: 1.5px solid {st_border}; border-radius: 12px; padding: 14px 20px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;'>
        <div style='display: flex; align-items: center; gap: 14px;'>
            <span style='font-size: 1.8rem;'>{sentiment_info.get('bias_icon', '📈')}</span>
            <div>
                <div style='font-size: 1.15rem; font-weight: 800; color: #f8fafc; font-family: "Outfit", sans-serif;'>{sentiment_info.get('title', 'Market Sentiment Ready')}</div>
                <div style='color: #cbd5e1; font-size: 0.85rem; margin-top: 2px;'>{sentiment_info.get('summary', 'Scanned pre-market cues.')}</div>
            </div>
        </div>
        <div>
            <span style='background: {st_bg}; color: {st_text_color}; border: 1px solid {st_border}; padding: 6px 14px; border-radius: 8px; font-weight: 800; font-size: 0.85rem; letter-spacing: 0.5px;'>{sentiment_info.get('sentiment', 'BULLISH')}</span>
        </div>
    </div>
    """).strip()
    st.markdown(sentiment_html, unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4)
    n_gap_sign = "+" if sentiment_info.get("gap_pct", 0) >= 0 else ""
    q1.metric("🇮🇳 NIFTY 50", f"₹{sentiment_info.get('nifty_price', 24200):,.2f}", f"{n_gap_sign}{sentiment_info.get('gap_pct', 0):.2f}% Gap", delta_color="normal")
    q2.metric("🏦 BANK NIFTY", f"₹{sentiment_info.get('banknifty_price', 51000):,.2f}")
    q3.metric("⚡ INDIA VIX (Volatility)", f"{sentiment_info.get('vix_level', 13.5):.2f}", "Normal Market" if sentiment_info.get('vix_level', 13.5) < 16 else "High Volatility")
    q4.metric("🎯 Total Suggestions Ready", f"{len(top_picks) + len(option_calls) + len(swing_picks)} Calls", "Stocks + Options + Swing")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Clean Segmented Navigation Tabs
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        f"⚡ Top Morning Stock Calls ({len(top_picks)})",
        f"🎯 Nifty & BankNifty Option Calls ({len(option_calls)})",
        f"💎 Positional & Swing Picks ({len(swing_picks)})",
        f"🔥 Gap & Volume Movers ({len(gap_ups) + len(gap_downs)})"
    ])

    # SUB-TAB 1: TOP MORNING STOCK CALLS
    with sub_tab1:
        st.markdown("<div style='margin-bottom: 12px; color: #94a3b8; font-size: 0.88rem;'>High-conviction equity breakout stocks scanned across 40+ liquid Indian companies. Filtered for Relative Strength and positive buyer volume.</div>", unsafe_allow_html=True)

        if top_picks:
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

                        card_html = textwrap.dedent(f"""
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
                        """).strip()
                        st.markdown(card_html, unsafe_allow_html=True)
                        
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

    # SUB-TAB 2: MORNING F&O OPTION CALLS
    with sub_tab2:
        st.markdown("<div style='margin-bottom: 12px; color: #94a3b8; font-size: 0.88rem;'>High-probability Index Option contracts with defined strike prices, option premiums in ₹, profit potential per lot, and disciplined stop-loss risk.</div>", unsafe_allow_html=True)

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

                    card_html = textwrap.dedent(f"""
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
                    """).strip()
                    st.markdown(card_html, unsafe_allow_html=True)

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

    # SUB-TAB 3: POSITIONAL & SWING PICKS
    with sub_tab3:
        st.markdown("<div style='margin-bottom: 12px; color: #94a3b8; font-size: 0.88rem;'>High-conviction multi-week wealth builder stocks for 2 to 4 week holding horizons with +6% to +12% target upside.</div>", unsafe_allow_html=True)

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

                    card_html = textwrap.dedent(f"""
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
                    """).strip()
                    st.markdown(card_html, unsafe_allow_html=True)

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

    # SUB-TAB 4: GAP & VOLUME MOVERS LEADERBOARD
    with sub_tab4:
        st.markdown("<div style='margin-bottom: 12px; color: #94a3b8; font-size: 0.88rem;'>Real-time morning momentum leaderboard showing stocks with the strongest opening gaps across the Indian market.</div>", unsafe_allow_html=True)

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
