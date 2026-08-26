"""
Autonomous AI Trading Agent & Auto-Pilot Daemon Tab.
"""

import os
import streamlit as st
import pandas as pd
import config
from src.utils.helpers import (
    get_ist_now, format_currency_inr, display_symbol_name, clean_symbol, format_holding_duration
)
from src.utils.storage import (
    load_ai_settings, save_ai_settings, get_portfolio_state, get_open_positions,
    get_calibration_records, get_disagreement_records
)
from src.data.data_fetcher import get_historical_data, get_live_quote
from src.ai import LLMClient, AITradingAgent, MarketRadarScanner
from src.ai.multi_agent_council import MultiAgentCouncil
from src.ai.autonomous_daemon import AutonomousAIDaemon
from src.engine.market_hunter_daemon import MarketHunterDaemon
from src.engine.ai_guardrails import AIGuardrails
from src.engine.reconciliation import StateReconciler
from src.engine.trade_manager import SmartTradeManager
from src.backtest.ai_backtester import AIBacktester

def render_autonomous_tab(broker_instance):
    """Renders the complete Autonomous AI Trading Agent & Auto-Pilot Tab."""
    saved_ai = load_ai_settings()
    saved_prov = saved_ai.get("provider", "gemini")
    
    prov_to_model = {
        "gemini": "gemini-3.7-flash",
        "groq": "llama-3.3-70b-versatile",
        "anthropic": "claude-3-7-sonnet-20250219",
        "openai": "gpt-4o",
        "deepseek": "deepseek-chat",
        "kimi": "moonshot-v1-8k",
        "ollama": "deepseek-r1:latest"
    }
    
    prov_key = saved_prov
    model_choice = saved_ai.get("model", prov_to_model.get(prov_key, "gemini-3.7-flash"))
    ai_api_key = saved_ai.get("api_key") or os.getenv(f"{prov_key.upper()}_API_KEY", "")
    
    active_ai_broker = broker_instance
    is_live_selected = False
    ai_guardrails = AIGuardrails(
        max_daily_loss_flat=2000.0,
        max_daily_loss_pct=3.0,
        max_concurrent_legs=2,
        max_lots_per_trade=1,
        min_confidence_threshold=7.5,
        sl_cooldown_minutes=15
    )
    
    llm_instance = LLMClient(provider=prov_key, model=model_choice, api_key=ai_api_key) if (ai_api_key or prov_key == "ollama") else None
    ai_daemon = AutonomousAIDaemon.get_instance()

    # 1. Master AI Auto-Pilot Banner & Control
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #0b1120 0%, #1e1b4b 100%); border: 2px solid #6366f1; border-radius: 12px; padding: 20px 24px; margin-bottom: 18px;'>
        <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;'>
            <div>
                <h2 style='margin: 0; color: #ffffff; font-size: 1.6rem;'>🤖 Autonomous AI Trading Bot</h2>
                <div style='color: #a5b4fc; font-size: 0.92rem; margin-top: 4px;'>
                    Zero manual steps required. The AI bot scans Indian markets (NSE/NFO), picks top trade setups, executes orders, manages stop-loss/targets, and books profits automatically.
                </div>
            </div>
            <div style='display: flex; gap: 10px; align-items: center;'>
                <span class='badge-bull'>🛡️ SEBI Risk Guardrails Active</span>
                <span class='badge-bear'>🛑 3:15 PM Auto Square-Off</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Master 1-Click Toggle & Real-Time Operational Telemetry
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2.5, 1.5, 1.5, 1.5])
    with ctrl_col1:
        if ai_daemon.is_active:
            if st.button("🛑 STOP AI AUTO-PILOT BOT", type="secondary", use_container_width=True):
                ai_daemon.stop()
                st.rerun()
        else:
            if st.button("🚀 START AI AUTO-PILOT BOT (1-CLICK)", type="primary", use_container_width=True):
                if not llm_instance or not llm_instance.is_configured():
                    st.info("💡 Tip: Enter your AI API Key in the settings below if you want custom LLM reasoning.")
                ai_daemon.start(
                    llm_client=llm_instance,
                    guardrails=ai_guardrails,
                    broker=active_ai_broker,
                    is_live_mode=is_live_selected,
                    interval=20
                )
                st.rerun()
                
    reconciled = StateReconciler.reconcile_with_broker(active_ai_broker)
    with ctrl_col2:
        st.metric("Bot Status", "🟢 ACTIVE & TRADING" if ai_daemon.is_active else "⚪ IDLE")
    with ctrl_col3:
        real_pnl = reconciled['daily_pnl']
        st.metric("Today's PnL", f"₹{real_pnl:+,.2f}", f"{'+' if real_pnl>=0 else ''}{real_pnl:.2f} ₹", delta_color="normal")
    with ctrl_col4:
        st.metric("Auto-Trades Executed", f"{ai_daemon.trades_executed_today}")
        
    st.markdown("---")

    # 3. Live AI Internal Thought Stream
    @st.fragment(run_every=2)
    def render_live_thought_feed():
        d_inst = AutonomousAIDaemon.get_instance()
        thoughts = d_inst.get_thought_stream()
        if thoughts:
            st.markdown("**🧠 Live AI Thought & Action Stream** *(Real-time internal reasoning)*")
            thought_html_lines = []
            for t in thoughts[:12]:
                lvl = t.get("level", "INFO")
                color = "#10b981" if lvl == "EXECUTE" else "#38bdf8" if lvl == "SETUP" else "#f43f5e" if lvl == "RISK" else "#f59e0b" if lvl == "EXIT" else "#a855f7" if lvl == "MANAGEMENT" else "#94a3b8"
                sym_tag = f"[{t.get('symbol')}]" if t.get('symbol') else ""
                thought_html_lines.append(
                    f"<div style='margin-bottom: 4px; line-height: 1.4;'><span style='color: #64748b; font-size: 0.78rem;'>{t.get('time')}</span> "
                    f"<span style='color: {color}; font-weight: 700; font-size: 0.78rem;'>[{lvl}]</span> "
                    f"<span style='color: #cbd5e1; font-weight: 600; font-size: 0.82rem;'>{sym_tag}</span> "
                    f"<span style='color: #f1f5f9; font-size: 0.84rem;'>{t.get('message')}</span></div>"
                )
            st.markdown(
                f"""<div style='background: #090d16; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 16px; font-family: "JetBrains Mono", monospace; max-height: 200px; overflow-y: auto;'>
                {''.join(thought_html_lines)}
                </div>""",
                unsafe_allow_html=True
            )
            
    render_live_thought_feed()

    # 4. Live Active Positions Feed
    @st.fragment(run_every=2)
    def render_active_positions_feed():
        open_pos = active_ai_broker.get_open_positions()
        if open_pos:
            st.markdown("**📊 Currently Active AI Trades (Auto-Managed):**")
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
                <div style='background: #111622; border-left: 5px solid {p_border}; border-top: 1px solid #1e293b; border-right: 1px solid #1e293b; border-bottom: 1px solid #1e293b; border-radius: 8px; padding: 12px 16px; margin-top: 8px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;'>
                        <div style='display: flex; align-items: center; gap: 10px;'>
                            <strong style='color: #f8fafc; font-size: 1.05rem; font-family: "Outfit", sans-serif;'>{p_sym}</strong>
                            <span class='badge-cyan' style='font-size: 0.72rem;'>{p_side} ({pos['quantity']} sh)</span>
                            <span class='badge-bull' style='font-size: 0.72rem;'>⏳ Holding: {p_dur}</span>
                        </div>
                        <div style='display: flex; align-items: center; gap: 14px;'>
                            <div style='font-size: 0.84rem; color: #94a3b8;'>Entry: ₹{p_entry:,.2f} &bull; LTP: ₹{p_curr:,.2f}</div>
                            <div style='font-size: 1.05rem; font-weight: 700; color: {p_border}; font-family: "JetBrains Mono", monospace;'>{format_currency_inr(p_pnl)} ({p_arr}{p_pnl_pct:.2f}%)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("🚨 Square Off All Active Positions (Clean Slate)", key="btn_sq_off_ai_daemon", type="secondary", use_container_width=True):
                active_ai_broker.square_off_all(reason="Manual Clean Slate Reset")
                st.success("✅ All positions squared off! Margin restored.")
                st.rerun()
                
    render_active_positions_feed()

    st.markdown("---")

    # 5. AI Market Opportunity Radar
    st.subheader("📡 Live AI Trade Setups (Auto-Dispatched on Auto-Pilot)")
    st.caption("The AI scans NIFTY, BANK NIFTY, and top stocks to calculate exact Buy Entry, Stop-Loss Exit, and Target Exit prices.")
    
    rad_c1, rad_c2, rad_c3 = st.columns([2, 1.6, 1.2])
    with rad_c2:
        radar_mode = st.selectbox(
            "Filter Threshold",
            ["High Probability (≥ 6.5)", "Breakouts Only (≥ 7.0)", "Conservative (≥ 7.5)", "Dips & Pullbacks (≥ 6.0)"],
            index=0,
            key="radar_threshold_select"
        )
        conf_thresh = 6.5 if "6.5" in radar_mode else (7.0 if "7.0" in radar_mode else (7.5 if "7.5" in radar_mode else 6.0))
    with rad_c1:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Scan Market Opportunities Now", type="secondary", use_container_width=True):
            with st.spinner("Scanning Indian markets for top institutional setups..."):
                radar_res = MarketRadarScanner.scan_market(llm_client=llm_instance, min_confidence=conf_thresh, force_refresh=True)
                st.session_state["last_radar_scan"] = radar_res
                st.rerun()
    with rad_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.caption(f"🧠 Engine: **{prov_key.upper()}** ({model_choice})")
        
    if "last_radar_scan" not in st.session_state:
        radar_res = MarketRadarScanner.scan_market(llm_client=llm_instance, min_confidence=6.5)
        st.session_state["last_radar_scan"] = radar_res

    @st.fragment(run_every=3)
    def render_opportunity_cards_fragment():
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
                        o_horizon = opp.get("time_horizon", "Exit by 3:15 PM IST")
                        o_setup = opp.get("setup_name", "Momentum Breakout")
                        o_reason = opp.get("catalyst_reasoning", "")
                        o_exp_str = opp.get("expiry_str", "Current Weekly")
                        o_lot = int(opp.get("lot_size", 75))
                        
                        is_opt = o_contract != "N/A" and ("CE" in o_contract or "PE" in o_contract)
                        u_quote = get_live_quote(o_sym)
                        u_spot = float(u_quote.get("price", opp.get("spot_price", 24250.0)))
                        u_chg = float(u_quote.get("change_pct", opp.get("spot_change_pct", 0.0)))
                        
                        if is_opt:
                            c_quote = get_live_quote(o_contract)
                            live_p = float(c_quote.get("price", opp.get("entry_price", 132.0)))
                            entry_val = live_p
                            sl_val = round(entry_val * 0.78, 1)
                            t1_val = round(entry_val * 1.35, 1)
                            t2_val = round(entry_val * 1.65, 1)
                            cap_val = round(entry_val * o_lot, 2)
                            is_bull = "CALL" in o_act or "CE" in o_contract
                            sp_trig_val = float(opp.get("spot_trigger", round(u_spot + (10.0 if is_bull else -10.0), 1)))
                            max_risk_inr = round(abs(entry_val - sl_val) * o_lot, 0)
                            exp_gain_inr = round(abs(t1_val - entry_val) * o_lot, 0)
                            
                            if is_bull:
                                trigger_met = u_spot >= sp_trig_val
                                pts_away = sp_trig_val - u_spot
                                zone_text = f"🟢 TRIGGER ACTIVE (Spot ≥ ₹{sp_trig_val:,.0f})" if trigger_met else f"⏳ AWAITING TRIGGER ({pts_away:.1f} pts away)"
                                zone_color = "#10b981" if trigger_met else "#f59e0b"
                            else:
                                trigger_met = u_spot <= sp_trig_val
                                pts_away = u_spot - sp_trig_val
                                zone_text = f"🟢 BREAKDOWN ACTIVE (Spot ≤ ₹{sp_trig_val:,.0f})" if trigger_met else f"⏳ AWAITING BREAKDOWN ({pts_away:.1f} pts away)"
                                zone_color = "#10b981" if trigger_met else "#f59e0b"
                        else:
                            live_p = u_spot
                            entry_val = float(opp.get("entry_price", u_spot))
                            sl_val = float(opp.get("stop_loss", u_spot * 0.985))
                            t1_val = float(opp.get("target_1", u_spot * 1.02))
                            t2_val = float(opp.get("target_2", u_spot * 1.04))
                            cap_val = round(entry_val * o_lot, 2)
                            sp_trig_val = entry_val
                            max_risk_inr = round(abs(entry_val - sl_val) * o_lot, 0)
                            exp_gain_inr = round(abs(t1_val - entry_val) * o_lot, 0)
                            is_bull = "BUY" in o_act
                            diff_pct = ((live_p - entry_val) / max(0.01, entry_val)) * 100.0 if entry_val > 0 else 0.0
                            if abs(diff_pct) <= 1.0:
                                zone_text = "🟢 READY TO ENTER"
                                zone_color = "#10b981"
                            elif diff_pct > 1.0:
                                zone_text = f"⚡ RUNNING (+{diff_pct:.1f}%)"
                                zone_color = "#38bdf8"
                            else:
                                zone_text = f"⏳ DISCOUNT ({diff_pct:.1f}%)"
                                zone_color = "#94a3b8"
                        
                        card_border = "#10b981" if is_bull else "#f43f5e"
                        contract_title = o_contract if is_opt else o_sym
                        action_badge_bg = "rgba(16, 185, 129, 0.15)" if is_bull else "rgba(244, 63, 94, 0.15)"
                        action_badge_color = "#10b981" if is_bull else "#f43f5e"
                        
                        sl_pct_calc = abs(round(((entry_val - sl_val) / max(0.01, entry_val)) * 100, 1))
                        t1_pct_calc = abs(round(((t1_val - entry_val) / max(0.01, entry_val)) * 100, 1))
                        t2_pct_calc = abs(round(((t2_val - entry_val) / max(0.01, entry_val)) * 100, 1))
                        
                        search_helper_html = ""
                        if is_opt:
                            u_search = opp.get("universal_search") or f"{o_sym} {o_contract}"
                            t_sym = opp.get("trading_symbol") or o_contract
                            m_ness = opp.get("moneyness", "ATM")
                            search_helper_html = f"""
                            <div style='background: rgba(15, 23, 42, 0.85); border: 1px dashed #38bdf8; border-radius: 8px; padding: 8px 14px; margin-bottom: 14px; font-size: 0.82rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;'>
                            <div>
                                <span style='color: #94a3b8;'>🔍 <strong>Broker Search (Zerodha / Groww / Dhan):</strong></span> 
                                <code style='background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 3px 8px; border-radius: 4px; font-weight: 800; font-size: 0.88rem;'>{u_search}</code>
                                <span style='color: #64748b; font-size: 0.78rem;'> &bull; Exchange Tradingsymbol: <code style='color: #cbd5e1;'>{t_sym}</code></span>
                            </div>
                            <div>
                                <span style='background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 3px 8px; border-radius: 4px; font-size: 0.76rem; font-weight: 700;'>📍 {m_ness}</span>
                            </div>
                            </div>
                            """
                        
                        card_html = f"""
                        <div style='background: linear-gradient(135deg, #0d121f 0%, #111827 100%); border: 1px solid #1e293b; border-left: 5px solid {card_border}; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 12px; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;'>
                        <div>
                        <span style='font-size: 1.25rem; font-weight: 800; color: #ffffff; font-family: "Outfit", sans-serif;'>#{o_rank} {contract_title}</span>
                        <span style='background: {action_badge_bg}; color: {action_badge_color}; border: 1px solid {action_badge_color}40; font-size: 0.78rem; font-weight: 700; padding: 3px 10px; border-radius: 6px; margin-left: 8px;'>{o_act}</span>
                        </div>
                        <div style='display: flex; gap: 10px; align-items: center;'>
                        <span style='color: #94a3b8; font-size: 0.82rem;'>📅 Expiry: <strong style='color: #f1f5f9;'>{o_exp_str}</strong></span>
                        <span style='background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); font-size: 0.78rem; font-weight: 700; padding: 3px 10px; border-radius: 6px;'>⭐ {o_conf:.1f}/10 Conviction</span>
                        </div>
                        </div>
                        {search_helper_html}
                        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 14px;'>
                        <div style='background: #090d16; border: 1px solid #38bdf8; border-radius: 8px; padding: 12px; text-align: center;'>
                        <div style='color: #38bdf8; font-size: 0.74rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;'>🟢 BUY ENTRY PRICE</div>
                        <div style='color: #ffffff; font-size: 1.45rem; font-weight: 900; font-family: "JetBrains Mono", monospace; margin: 4px 0;'>₹{entry_val:,.2f}</div>
                        <div style='color: {zone_color}; font-size: 0.75rem; font-weight: 700;'>{zone_text}</div>
                        </div>
                        <div style='background: #090d16; border: 1px solid rgba(244, 63, 94, 0.5); border-radius: 8px; padding: 12px; text-align: center;'>
                        <div style='color: #f43f5e; font-size: 0.74rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;'>🛑 STOP-LOSS EXIT</div>
                        <div style='color: #f43f5e; font-size: 1.45rem; font-weight: 900; font-family: "JetBrains Mono", monospace; margin: 4px 0;'>₹{sl_val:,.2f}</div>
                        <div style='color: #f43f5e; font-size: 0.75rem; font-weight: 700;'>Exit on -{sl_pct_calc}% (-₹{max_risk_inr:,.0f})</div>
                        </div>
                        <div style='background: #090d16; border: 1px solid rgba(16, 185, 129, 0.5); border-radius: 8px; padding: 12px; text-align: center;'>
                        <div style='color: #10b981; font-size: 0.74rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;'>🎯 TARGET 1 (50%)</div>
                        <div style='color: #10b981; font-size: 1.45rem; font-weight: 900; font-family: "JetBrains Mono", monospace; margin: 4px 0;'>₹{t1_val:,.2f}</div>
                        <div style='color: #10b981; font-size: 0.75rem; font-weight: 700;'>Book 50% (+{t1_pct_calc}% / +₹{exp_gain_inr:,.0f})</div>
                        </div>
                        <div style='background: #090d16; border: 1px solid rgba(16, 185, 129, 0.5); border-radius: 8px; padding: 12px; text-align: center;'>
                        <div style='color: #10b981; font-size: 0.74rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;'>🚀 TARGET 2 (100%)</div>
                        <div style='color: #10b981; font-size: 1.45rem; font-weight: 900; font-family: "JetBrains Mono", monospace; margin: 4px 0;'>₹{t2_val:,.2f}</div>
                        <div style='color: #10b981; font-size: 0.75rem; font-weight: 700;'>Full Exit (+{t2_pct_calc}% / +₹{round(abs(t2_val-entry_val)*o_lot, 0):,.0f})</div>
                        </div>
                        </div>
                        <div style='display: flex; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; border-radius: 6px; padding: 8px 14px; font-size: 0.82rem; color: #cbd5e1; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;'>
                        <div>📦 <strong>Position:</strong> 1 Lot ({o_lot} Shares) &bull; Total Capital: <strong style='color: #38bdf8;'>₹{cap_val:,.0f}</strong></div>
                        <div>🇮🇳 <strong>{o_sym.replace('^','')} Spot:</strong> ₹{u_spot:,.2f} (<span style='color: {"#10b981" if u_chg >= 0 else "#f43f5e"};'>{'+' if u_chg >= 0 else ''}{u_chg:.2f}%</span>) &bull; <strong>Trigger:</strong> Spot {'≥' if is_bull else '≤'} ₹{sp_trig_val:,.1f}</div>
                        <div>⏱️ <strong>Holding Duration:</strong> {o_horizon}</div>
                        </div>
                        <div style='color: #94a3b8; font-size: 0.84rem; line-height: 1.4;'>
                        🧠 <strong style='color: #e2e8f0;'>Strategy:</strong> {o_setup} &mdash; {o_reason}
                        </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        if st.button(f"🚀 1-Click Execute #{o_rank}: Buy {contract_title} @ ₹{live_p:,.1f} (₹{cap_val:,.0f})", key=f"btn_exec_opp_{i}", type="primary", use_container_width=True):
                            llm_inst = LLMClient(provider=prov_key, model=model_choice, api_key=ai_api_key) if (ai_api_key and len(ai_api_key.strip()) >= 5) else None
                            agent = AITradingAgent(
                                llm_client=llm_inst,
                                guardrails=ai_guardrails,
                                broker=active_ai_broker,
                                is_live_mode=is_live_selected
                            )
                            opp_exec = dict(opp)
                            opp_exec["entry_price"] = live_p
                            exec_outcome = agent.execute_radar_opportunity(opp_exec)
                            if exec_outcome.get("status") == "EXECUTED":
                                st.success(f"✅ Trade Executed! Symbol: `{exec_outcome.get('symbol')}` @ ₹{live_p:,.2f}")
                                st.rerun()
                            else:
                                st.error(f"❌ Execution Blocked: {exec_outcome.get('message')}")
                        st.write("")
            elif r_data.get("status") == "ERROR":
                st.error(f"❌ {r_data.get('message')}")
                
    render_opportunity_cards_fragment()

    st.markdown("---")
    st.subheader("🧠 Multi-Agent AI Strategy Council & Autonomous Market Hunter")
    st.caption("3 specialized orthogonal agents evaluate candidate breakouts with 2-stage gating and software-managed OCO execution (entry + standalone exchange SL-M order).")
    
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

            bp = c_res.get("trade_blueprint", {})
            if bp:
                bp_act = bp.get("action", "BUY")
                bp_entry = float(bp.get("entry_price", c_res.get("current_price", 100.0)))
                bp_sl = float(bp.get("stop_loss_price", bp_entry * 0.985))
                bp_sl_pct = float(bp.get("stop_loss_pct", 1.5))
                bp_t1 = float(bp.get("target_1_price", bp_entry * 1.025))
                bp_t1_pct = float(bp.get("target_1_gain_pct", 2.5))
                bp_t2 = float(bp.get("target_2_price", bp_entry * 1.050))
                bp_t2_pct = float(bp.get("target_2_gain_pct", 5.0))
                bp_qty = max(1, int(25000.0 / max(1.0, bp_entry)))
                bp_cap = round(bp_entry * bp_qty, 2)
                bp_risk_inr = round(abs(bp_entry - bp_sl) * bp_qty, 0)
                bp_gain_inr = round(abs(bp_t1 - bp_entry) * bp_qty, 0)

                council_bp_html = f"""
                <div style='background: #090d16; border: 1px solid #1e293b; border-radius: 10px; padding: 14px 18px; margin-top: 14px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <div style='color: #38bdf8; font-size: 0.85rem; font-weight: 800; text-transform: uppercase;'>🎯 Council Actionable Trade Blueprint</div>
                <span class='{"badge-bull" if c_app else "badge-neutral"}'>R:R 1:2.0 &bull; MIS Intraday (3:15 PM Auto-Exit)</span>
                </div>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px;'>
                <div style='background: #111622; border: 1px solid #38bdf8; border-radius: 6px; padding: 10px; text-align: center;'>
                <div style='color: #38bdf8; font-size: 0.70rem; font-weight: 800;'>🟢 BUY ENTRY</div>
                <div style='color: #ffffff; font-size: 1.3rem; font-weight: 900; font-family: "JetBrains Mono", monospace;'>₹{bp_entry:,.2f}</div>
                <div style='color: #94a3b8; font-size: 0.72rem;'>{bp.get("entry_zone", f"₹{bp_entry:,.2f}")}</div>
                </div>
                <div style='background: #111622; border: 1px solid rgba(244, 63, 94, 0.5); border-radius: 6px; padding: 10px; text-align: center;'>
                <div style='color: #f43f5e; font-size: 0.70rem; font-weight: 800;'>🛑 SAFETY STOP-LOSS</div>
                <div style='color: #f43f5e; font-size: 1.3rem; font-weight: 900; font-family: "JetBrains Mono", monospace;'>₹{bp_sl:,.2f}</div>
                <div style='color: #f43f5e; font-size: 0.72rem;'>-{bp_sl_pct:.1f}% (-₹{bp_risk_inr:,.0f})</div>
                </div>
                <div style='background: #111622; border: 1px solid rgba(16, 185, 129, 0.5); border-radius: 6px; padding: 10px; text-align: center;'>
                <div style='color: #10b981; font-size: 0.70rem; font-weight: 800;'>🎯 TARGET 1 (50% LOCK)</div>
                <div style='color: #10b981; font-size: 1.3rem; font-weight: 900; font-family: "JetBrains Mono", monospace;'>₹{bp_t1:,.2f}</div>
                <div style='color: #10b981; font-size: 0.72rem;'>+{bp_t1_pct:.1f}% (+₹{bp_gain_inr:,.0f})</div>
                </div>
                <div style='background: #111622; border: 1px solid rgba(16, 185, 129, 0.5); border-radius: 6px; padding: 10px; text-align: center;'>
                <div style='color: #10b981; font-size: 0.70rem; font-weight: 800;'>🚀 TARGET 2 (RUNNER)</div>
                <div style='color: #10b981; font-size: 1.3rem; font-weight: 900; font-family: "JetBrains Mono", monospace;'>₹{bp_t2:,.2f}</div>
                <div style='color: #10b981; font-size: 0.72rem;'>+{bp_t2_pct:.1f}% (+₹{round(abs(bp_t2-bp_entry)*bp_qty, 0):,.0f})</div>
                </div>
                </div>
                </div>
                """
                st.markdown(council_bp_html, unsafe_allow_html=True)
                
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                if st.button(f"🚀 1-Click Place {bp_act} Order: {c_res.get('display_name')} @ ₹{bp_entry:,.2f} (₹{bp_cap:,.0f})", key=f"btn_council_order_{c_sym}", type="primary", use_container_width=True):
                    order_res = active_ai_broker.place_order(
                        symbol=c_sym,
                        side=bp_act,
                        quantity=bp_qty,
                        price=bp_entry,
                        sl=bp_sl,
                        tp=bp_t1,
                        strategy_name="MultiAgent_Council_Setup"
                    )
                    if order_res.get("status") in ["FILLED", "SUCCESS"]:
                        st.success(f"✅ Order Placed! {bp_act} {bp_qty} shares of {c_res.get('display_name')} @ ₹{bp_entry:,.2f}. SL: ₹{bp_sl:,.2f}, Target: ₹{bp_t1:,.2f}")
                        st.rerun()
                    else:
                        st.error(f"❌ Execution Blocked: {order_res.get('message')}")

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
    with st.expander("📊 **Multi-Regime Historical Stress Replay**", expanded=False):
        st.caption("Replay historical market regimes through the prompt and guardrail engine.")
        
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
