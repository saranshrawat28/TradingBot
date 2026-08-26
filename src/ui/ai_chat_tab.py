"""
Natural Language AI Chat & Agentic Voice/Text Assistant Tab.
Enhanced with:
1. Agentic Tool Calling & Portfolio Telemetry Actions.
2. Multi-Turn Context Memory.
3. Interactive Plotly Mini-Charts & Visual Proportional R:R Price Ladders.
"""

import streamlit as st
import plotly.graph_objects as go
import textwrap

from src.ai.chat_assistant import TradingChatAssistant
from src.engine.ai_guardrails import AIGuardrails
from src.utils.storage import load_ai_settings, get_portfolio_state
from src.utils.helpers import get_ist_now

def _render_mini_chart(chart_data: dict):
    """Renders a sleek, dark-themed interactive Plotly candlestick mini-chart."""
    if not chart_data or not chart_data.get("dates"):
        return

    sym = chart_data.get("display_name", chart_data.get("symbol", "Stock"))
    dates = chart_data.get("dates", [])
    open_p = chart_data.get("open", [])
    high_p = chart_data.get("high", [])
    low_p = chart_data.get("low", [])
    close_p = chart_data.get("close", [])
    ema9 = chart_data.get("ema9", [])
    ema21 = chart_data.get("ema21", [])

    entry_p = chart_data.get("entry_price", 0.0)
    t1_p = chart_data.get("target_1", 0.0)
    t2_p = chart_data.get("target_2", 0.0)
    sl_p = chart_data.get("stop_loss", 0.0)

    fig = go.Figure()

    # Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=dates,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        name="Price",
        increasing_line_color="#10b981",
        decreasing_line_color="#f43f5e"
    ))

    # EMA 9 & EMA 21 Traces
    if ema9:
        fig.add_trace(go.Scatter(x=dates, y=ema9, name="9 EMA", line=dict(color="#38bdf8", width=1.2)))
    if ema21:
        fig.add_trace(go.Scatter(x=dates, y=ema21, name="21 EMA", line=dict(color="#fbbf24", width=1.2)))

    # Horizontal Overlay Lines
    poc_p = chart_data.get("poc", 0.0)
    if poc_p > 0:
        fig.add_hline(y=poc_p, line_dash="dot", line_color="#c084fc", line_width=1.2, annotation_text="POC", annotation_position="top left")
    if entry_p > 0:
        fig.add_hline(y=entry_p, line_dash="solid", line_color="#38bdf8", line_width=1.5, annotation_text="Entry", annotation_position="top right")
    if t1_p > 0:
        fig.add_hline(y=t1_p, line_dash="dash", line_color="#10b981", line_width=1.5, annotation_text="Target 1", annotation_position="top right")
    if t2_p > 0:
        fig.add_hline(y=t2_p, line_dash="dot", line_color="#059669", line_width=1.2, annotation_text="Target 2", annotation_position="top right")
    if sl_p > 0:
        fig.add_hline(y=sl_p, line_dash="dash", line_color="#f43f5e", line_width=1.5, annotation_text="Stop-Loss", annotation_position="bottom right")

    fig.update_layout(
        title=f"📊 {sym} — Live Price Structure & Execution Targets",
        title_font=dict(size=12, color="#94a3b8"),
        template="plotly_dark",
        paper_bgcolor="#080b11",
        plot_bgcolor="#0b0f19",
        height=270,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False), tickfont=dict(size=9, color="#64748b")),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", tickfont=dict(size=9, color="#94a3b8"), side="right"),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def _render_action_card(msg: dict, msg_idx: int, broker_instance):
    """Renders specialized interactive Action Cards (Trade, Square-Off, Options, Portfolio)."""
    card = msg.get("action_card")
    card_type = msg.get("ui_card_type", "TRADE")

    if not card:
        return

    # 1. SQUARE_OFF Action Card
    if card_type == "SQUARE_OFF":
        mode = card.get("mode", "SINGLE")
        disp = card.get("display_name", "Position")
        qty = card.get("quantity", 0)
        curr_p = card.get("current_price", 0.0)
        pnl = card.get("pnl", 0.0)
        pnl_pct = card.get("pnl_pct", 0.0)
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_col = "#10b981" if pnl >= 0 else "#f43f5e"

        st.markdown(textwrap.dedent(f"""
        <div style='background: #111622; border: 2px solid #f43f5e; border-radius: 10px; padding: 16px; margin: 12px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='font-size: 1.10rem; font-weight: 800; color: #f8fafc;'>🛑 Square-Off Confirmation: {disp}</div>
                <span style='background: #f43f5e22; color: #f43f5e; border: 1px solid #f43f5e; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;'>EXIT POSITION</span>
            </div>
            <div style='margin: 10px 0; font-size: 0.88rem; color: #cbd5e1;'>
                • Quantity to Close: <strong>{qty} Shares</strong> @ Market Price ₹{curr_p:,.2f}<br>
                • Estimated P&L: <strong style='color: {pnl_col};'>{pnl_sign}₹{pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)</strong>
            </div>
            <div style='color: #94a3b8; font-size: 0.78rem;'>Clicking below will execute market sell on your broker and release margin immediately.</div>
        </div>
        """).strip(), unsafe_allow_html=True)

        btn_key = f"chat_sqoff_btn_{msg_idx}"
        if st.button(f"🛑 Execute Square-Off for {disp}", key=btn_key, type="primary", use_container_width=True):
            if mode == "ALL":
                res = broker_instance.square_off_all(reason="Chat Square Off All")
                st.success(f"✅ All {card.get('count', 0)} open positions squared off successfully!")
            else:
                target_sym = card.get("target_symbol", "")
                res = broker_instance.square_off_position(target_sym, reason="Chat User Request")
                st.success(f"✅ Squared off {disp}! Released {qty} shares @ ₹{curr_p:,.2f}.")
            st.rerun()

    # 2. OPTIONS Greeks & Strike Card
    elif card_type == "OPTIONS":
        contract = card.get("contract_symbol", "OPTION")
        spot = card.get("spot_price", 0.0)
        prem = card.get("theoretical_premium", 0.0)
        lot = card.get("lot_size", 25)
        cap = card.get("capital_required", 0.0)
        t1_prem = card.get("target_1_premium", 0.0)
        sl_prem = card.get("stop_loss_premium", 0.0)
        greeks = card.get("greeks", {})

        st.markdown(textwrap.dedent(f"""
        <div style='background: #111622; border: 2px solid #38bdf8; border-radius: 10px; padding: 16px; margin: 12px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='font-size: 1.12rem; font-weight: 800; color: #f8fafc;'>🎯 {contract}</div>
                <span style='background: #38bdf822; color: #38bdf8; border: 1px solid #38bdf8; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;'>F&O SMART STRIKE</span>
            </div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0; font-size: 0.85rem;'>
                <div>💵 Est. Premium: <strong style='color: #38bdf8;'>₹{prem:,.2f}</strong></div>
                <div>📦 1 Lot ({lot} Qty): <strong>₹{cap:,.2f}</strong></div>
                <div>🎯 Target 1 (+35%): <strong style='color: #10b981;'>₹{t1_prem:,.2f}</strong></div>
                <div>🛑 Stop-Loss (-25%): <strong style='color: #f43f5e;'>₹{sl_prem:,.2f}</strong></div>
            </div>
            <div style='background: #080b11; border: 1px solid #1e293b; border-radius: 6px; padding: 8px 12px; margin-top: 8px; font-size: 0.78rem; display: flex; justify-content: space-between;'>
                <span>Δ Delta: <strong>{greeks.get('delta', 0.5):+.2f}</strong></span>
                <span>Θ Theta: <strong style='color: #f43f5e;'>₹{greeks.get('theta', -5.0):.2f}/day</strong></span>
                <span>Γ Gamma: <strong>{greeks.get('gamma', 0.001):.4f}</strong></span>
                <span>IV: <strong>{greeks.get('iv_pct', 15.0):.1f}%</strong></span>
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)

    # 3. TRADE Action Card with Visual Proportional R:R Price Ladder
    elif card_type == "TRADE":
        sym = card["symbol"]
        name = card["display_name"]
        action = card["action"]
        qty = card["quantity"]
        entry_p = card["entry_price"]
        cap = card["capital_required"]
        t1 = card["target_1_price"]
        t1_prof = card["target_1_profit"]
        t1_gain = card.get("target_1_gain_pct", 3.0)
        t2 = card.get("target_2_price", entry_p * 1.06)
        t2_prof = card.get("target_2_profit", (t2 - entry_p) * qty)
        t2_gain = card.get("target_2_gain_pct", 6.0)
        sl = card["stop_loss_price"]
        sl_risk = card["stop_loss_risk"]
        sl_pct = card.get("stop_loss_pct", 2.0)
        score = card["score"]
        act_badge = "#10b981" if action == "BUY" else "#f43f5e"

        st.markdown(textwrap.dedent(f"""<div style='background: #111622; border: 2px solid {act_badge}; border-radius: 10px; padding: 16px; margin: 12px 0;'>
<div style='display: flex; justify-content: space-between; align-items: center;'>
<div style='font-size: 1.15rem; font-weight: 800; color: #f8fafc;'>{name}</div>
<span style='background: rgba(16, 185, 129, 0.15); color: {act_badge}; border: 1px solid {act_badge}; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;'>{action} &bull; {card["product_type"]}</span>
</div>
<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; font-size: 0.85rem;'>
<div>📦 Quantity: <strong style='color: #f8fafc;'>{qty} Shares</strong></div>
<div>💵 Capital Required: <strong style='color: #f8fafc;'>₹{cap:,.2f}</strong></div>
<div>🎯 Target 1: <strong style='color: #10b981;'>₹{t1:,.2f} (+₹{t1_prof:,.2f})</strong></div>
<div>🛑 Safety SL: <strong style='color: #f43f5e;'>₹{sl:,.2f} (-₹{sl_risk:,.2f})</strong></div>
</div>
<div style='background: #080b11; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; margin: 10px 0;'>
<div style='display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 4px;'>
<span style='color: #f43f5e;'>🛑 SL: ₹{sl:,.2f} (-{sl_pct:.1f}%)</span>
<span style='color: #38bdf8;'>📍 Entry: ₹{entry_p:,.2f}</span>
<span style='color: #10b981;'>🎯 T1: ₹{t1:,.2f} (+{t1_gain:.1f}%)</span>
<span style='color: #059669;'>🚀 T2: ₹{t2:,.2f} (+{t2_gain:.1f}%)</span>
</div>
<div style='height: 6px; width: 100%; border-radius: 3px; display: flex; overflow: hidden;'>
<div style='background: #f43f5e; width: 25%;'></div>
<div style='background: #38bdf8; width: 25%;'></div>
<div style='background: #10b981; width: 25%;'></div>
<div style='background: #059669; width: 25%;'></div>
</div>
</div>
<div style='color: #94a3b8; font-size: 0.78rem;'>AI Mathematical Score: <strong style='color: #f8fafc;'>{score:.1f} / 10.0</strong> &bull; Zero-Bypass Guardrails Engaged</div>
</div>""").strip(), unsafe_allow_html=True)
        
        btn_key = f"chat_trade_btn_{sym}_{msg_idx}"
        if st.button(f"🚀 Confirm & Place {action} Order (₹{cap:,.0f})", key=btn_key, type="primary", use_container_width=True):
            proposal = {
                "symbol": sym,
                "target_asset": sym,
                "action": "BUY_STOCK" if action == "BUY" else "SELL_STOCK",
                "confidence_score": score,
                "entry_price": entry_p,
                "sl": sl,
                "target_1": t1,
                "horizon": "intraday",
                "notes": f"Chat Order for {name}"
            }
            
            p_state = get_portfolio_state()
            guard = AIGuardrails(min_confidence_threshold=7.0)
            approved, g_reason, sanitized_order = guard.evaluate_proposal(proposal, p_state, enforce_time_cutoff=True)
            
            if approved:
                order_res = broker_instance.place_order(
                    symbol=sym,
                    side=action,
                    quantity=qty,
                    price=entry_p,
                    sl=sl,
                    tp=t1,
                    strategy_name="Chat_Assistant_Order"
                )
                if order_res.get("status") in ["FILLED", "SUCCESS"]:
                    st.success(f"✅ Order Executed! Bought {qty} shares of {name} @ ₹{entry_p:,.2f}. Safety SL set @ ₹{sl:,.2f}.")
                    st.rerun()
                else:
                    st.error(f"❌ Order Failed: {order_res.get('message')}")
            else:
                st.error(f"🛡️ Guardrail Protected: {g_reason}")

def render_chat_assistant_tab(broker_instance):
    """Renders the conversational AI assistant with Agentic Tools & Multi-turn Memory."""
    st.markdown("""
    <div style='margin-bottom: 8px;'>
        <h2 style='margin: 0; font-family: "Outfit", sans-serif;'>🗣️ Talk to Your ApexTrade AI Bot</h2>
        <div style='color: #94a3b8; font-size: 0.92rem; margin-top: 4px;'>
            Ask anything in plain English or Hinglish — analyze stocks with live mini-charts, check portfolio telemetry, select F&O strikes, and execute safe bracket trades.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize State & Memory
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = {}

    saved_ai = load_ai_settings()
    has_llm = saved_ai.get("is_connected") and saved_ai.get("api_key")
    
    if has_llm:
        st.markdown(f"""
        <div style='background: #111622; border: 1px solid #10b981; border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: #10b981; font-weight: 700; font-size: 0.88rem;'><span class='ambient-dot-green'></span>AI BRAIN ACTIVE: {saved_ai['provider'].upper()} ({saved_ai.get('model', 'gemini-3.1-flash-lite')})</span>
            <span class='badge-bull'>AGENTIC TOOLS & CONTEXT MEMORY ENGAGED</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: #111622; border: 1px solid #f59e0b; border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: #f59e0b; font-weight: 700; font-size: 0.88rem;'>⚡ LOCAL HEURISTIC MODE (Deterministic Rule Engine & Zero Hallucinations)</span>
            <span class='badge-neutral'>CONNECT API IN SETTINGS FOR LLM</span>
        </div>
        """, unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "👋 **Hello! I am your ApexTrade AI Assistant.**\n\nI can analyze Indian stocks with live mini-charts, inspect your portfolio, select optimal F&O option strikes, or propose safe trades with automatic stop-loss.\n\n*Try one of the quick suggestions below or type your question!*",
                "action_card": None,
                "ui_card_type": None,
                "chart_data": None,
                "timestamp": get_ist_now().strftime("%I:%M %p")
            }
        ]

    # Quick Suggestion Chips
    st.markdown("<div style='font-size: 0.80rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;'>💡 Quick Ideas:</div>", unsafe_allow_html=True)
    q_c1, q_c2, q_c3, q_c4, q_c5 = st.columns(5)
    selected_quick_query = None
    with q_c1:
        if st.button("🌅 Market Opening Mood", use_container_width=True):
            selected_quick_query = "What is the market opening mood today?"
    with q_c2:
        if st.button("📊 Analyze Tata Motors", use_container_width=True):
            selected_quick_query = "How is Tata Motors looking for intraday?"
    with q_c3:
        if st.button("🎯 Nifty Weekly Strike", use_container_width=True):
            selected_quick_query = "Suggest a Nifty Call option strike for weekly expiry"
    with q_c4:
        if st.button("💼 My Balance & P&L", use_container_width=True):
            selected_quick_query = "What is my account balance and profit today?"
    with q_c5:
        if st.button("🚀 Buy ₹25,000 Reliance", use_container_width=True):
            selected_quick_query = "Buy ₹25,000 of Reliance with safety stop-loss"

    # Chat Display Loop
    for msg_idx, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            
            # Render Mini-Chart if attached
            if msg.get("chart_data"):
                _render_mini_chart(msg["chart_data"])
                
            # Render Action Card if attached
            if msg.get("action_card"):
                _render_action_card(msg, msg_idx, broker_instance)

    user_input = st.chat_input("Ask about Indian stocks, follow-up on previous stock, or say 'Square off Reliance'...")
    final_query = selected_quick_query or user_input

    if final_query:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": final_query,
            "action_card": None,
            "ui_card_type": None,
            "chart_data": None,
            "timestamp": get_ist_now().strftime("%I:%M %p")
        })

        with st.spinner("ApexTrade AI is processing query..."):
            res = TradingChatAssistant.process_query(
                user_query=final_query,
                chat_history=[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages[-6:]],
                provider=saved_ai.get("provider", "gemini"),
                api_key=saved_ai.get("api_key"),
                model=saved_ai.get("model"),
                broker_instance=broker_instance,
                active_context=st.session_state.get("chat_context", {})
            )
            
            # Update Context Memory
            if res.get("updated_context"):
                st.session_state.chat_context = res["updated_context"]

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": res["response_text"],
                "action_card": res.get("action_card"),
                "ui_card_type": res.get("ui_card_type"),
                "chart_data": res.get("chart_data"),
                "timestamp": res.get("timestamp", get_ist_now().strftime("%I:%M %p"))
            })
            st.rerun()
