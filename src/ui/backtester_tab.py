"""
Strategy Backtester & Optimizer Tab.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data.data_fetcher import get_historical_data, get_live_quote, search_indian_stocks
from src.strategies import get_strategy
from src.engine.risk_manager import RiskManager
from src.engine.backtester import Backtester
from src.utils.helpers import display_symbol_name, format_currency_inr

def render_backtester_tab(broker_instance):
    """Renders the Strategy Backtester & Historical Simulation Engine."""
    st.markdown("""
    <h2>📊 Strategy Backtester — Test Any Indian Stock</h2>
    <div style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 14px;'>Test how much money a strategy would have made on any Indian stock in the past with realistic taxes, brokerage, and safety rules.</div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔍 **Step 1: Search & Select Indian Stock to Test**", expanded=True):
        bt_col1, bt_col2 = st.columns([2.5, 2])
        with bt_col1:
            bt_search = st.text_input("🔍 Search Company Name or Ticker to Backtest:", value="", placeholder="Type any name e.g. Tata, Zomato, Reliance, Suzlon, SBI, Paytm, HAL, ITC...")
            matching_bt = search_indian_stocks(bt_search)
            
            selected_sym = st.selectbox(
                f"Choose from Matching Results ({len(matching_bt)} available):",
                options=[item["symbol"] for item in matching_bt],
                format_func=lambda s: next((f"{item['name']} ({item['symbol'].replace('.NS','')}) — {item.get('category', 'Equity')}" for item in matching_bt if item["symbol"] == s), s),
                index=0
            )
            
            bt_quote = get_live_quote(selected_sym)
            if bt_quote.get("price", 0) > 0:
                st.caption(f"⚡ **Live LTP:** ₹{bt_quote['price']:,.2f} ({bt_quote['change_pct']:+.2f}%) | Previous Close: ₹{bt_quote['previous_close']:,.2f}")
        with bt_col2:
            st.info("💡 **Tip:** You can test any Indian stock across 5 different technical strategies to see historical profits, win rate %, and risk.")
                
    with st.expander("🧠 **Step 2: Choose Strategy & Simple Settings**", expanded=True):
        st_col1, st_col2 = st.columns([2, 2])
        with st_col1:
            strategy_choice = st.selectbox(
                "Select Trading Strategy:",
                [
                    "🚀 Trend Rider (EMA Crossover + RSI)",
                    "🛡️ Smart Safety Net (SuperTrend)",
                    "⚡ Discount Dip Buyer (Bollinger Bands)",
                    "🌊 Momentum Wave (MACD)",
                    "🎯 All-Rounder Combo (Multi-Indicator Confluence)"
                ]
            )
            
            if "Trend Rider" in strategy_choice:
                internal_strat_name = "EMA Crossover + RSI"
                st.markdown("""
                <div class='tip-box'>
                    💡 <strong>How Trend Rider Works:</strong><br>
                    • <strong>BUY:</strong> When the short-term trend crosses above the long-term trend AND buyer energy is strong.<br>
                    • <strong>SELL:</strong> When the trend cools down or momentum reverses.
                </div>
                """, unsafe_allow_html=True)
            elif "Smart Safety Net" in strategy_choice:
                internal_strat_name = "SuperTrend (Intraday/Swing)"
                st.markdown("""
                <div class='tip-box'>
                    💡 <strong>How Smart Safety Net Works:</strong><br>
                    • <strong>BUY:</strong> Enters when the SuperTrend indicator turns <strong>GREEN</strong>.<br>
                    • <strong>SELL:</strong> Exits immediately when the line turns <strong>RED</strong> to protect capital.
                </div>
                """, unsafe_allow_html=True)
            elif "Discount Dip Buyer" in strategy_choice:
                internal_strat_name = "Bollinger Bands"
                st.markdown("""
                <div class='tip-box'>
                    💡 <strong>How Discount Dip Buyer Works:</strong><br>
                    • <strong>BUY:</strong> When the stock price drops below its normal range (on sale) and begins bouncing back up.<br>
                    • <strong>SELL:</strong> When the price reaches the top expensive band.
                </div>
                """, unsafe_allow_html=True)
            elif "Momentum Wave" in strategy_choice:
                internal_strat_name = "MACD Momentum"
                st.markdown("""
                <div class='tip-box'>
                    💡 <strong>How Momentum Wave Works:</strong><br>
                    • <strong>BUY:</strong> Catches the start of a fast rally when momentum accelerates upward.<br>
                    • <strong>SELL:</strong> Exits when momentum slows down.
                </div>
                """, unsafe_allow_html=True)
            else:
                internal_strat_name = "Multi-Indicator Confluence"
                st.markdown("""
                <div class='tip-box'>
                    💡 <strong>How All-Rounder Combo Works (Recommended):</strong><br>
                    • <strong>BUY:</strong> Highest safety — only buys when Trend, Speed (RSI), Momentum (MACD), and Volatility (SuperTrend) <strong>ALL agree</strong>.
                </div>
                """, unsafe_allow_html=True)
                
        with st_col2:
            tf_col, per_col = st.columns(2)
            with tf_col:
                timeframe = st.selectbox(
                    "Candle Timeframe:",
                    ["5m (Day Trading)", "15m (Short-Term)", "1h (Swing Trading)", "1d (Positional / Days)"],
                    index=1
                )
                tf_clean = timeframe.split(" ")[0]
            with per_col:
                history_period = st.selectbox(
                    "Test Duration:",
                    ["1mo (Past Month)", "3mo (Past 3 Months)", "6mo (Past 6 Months)", "1y (Past 1 Year)", "2y (Past 2 Years)"],
                    index=2
                )
                per_clean = history_period.split(" ")[0]
                
            st.markdown("##### 🛡️ Safety & Money Settings")
            r1, r2, r3 = st.columns(3)
            with r1:
                test_capital = st.number_input("Starting Capital (₹)", value=100000.0, step=10000.0)
            with r2:
                sl_pct = st.slider("Safety Stop-Loss % (Max Loss)", 0.5, 5.0, 1.5, 0.25, help="If the trade loses this %, the bot cuts the loss automatically.")
            with r3:
                tp_pct = st.slider("Profit Target % (Goal)", 1.0, 10.0, 3.0, 0.5, help="When the trade reaches this profit %, the bot sells to pocket the gain.")
                
    run_btn = st.button("🚀 Run Backtest Simulation Now", type="primary", use_container_width=True)
    
    if run_btn:
        with st.spinner(f"Loading real market data for {display_symbol_name(selected_sym)} and running test..."):
            hist_df = get_historical_data(selected_sym, period=per_clean, interval=tf_clean)
            
            if hist_df.empty:
                st.error(f"Could not load data for {selected_sym}. Please verify the symbol or choose a different timeframe.")
            else:
                strategy = get_strategy(internal_strat_name)
                risk_mgr = RiskManager(default_sl_pct=sl_pct, default_tp_pct=tp_pct)
                backtester = Backtester(strategy=strategy, initial_capital=test_capital, risk_manager=risk_mgr)
                results = backtester.run(hist_df)
                
                st.markdown("### 🏆 Easy Performance Scorecard")
                k1, k2, k3, k4, k5 = st.columns(5)
                
                net_p = results["net_profit"]
                ret_p = results["total_return_pct"]
                ret_arr = "▲ +" if ret_p >= 0 else "▼ "
                
                k1.metric("💰 Total Profit / Loss", format_currency_inr(net_p), f"{ret_arr}{ret_p:.2f}%", delta_color="normal")
                k2.metric("🎯 Win Score", f"{results['win_rate_pct']:.1f}%", f"{results['winning_trades']} Won / {results['losing_trades']} Lost")
                k3.metric("📊 Nifty 50 Comparison", f"{results['benchmark_return_pct']:+.2f}%", f"Strategy: {ret_p:+.2f}%")
                k4.metric("🛡️ Max Drop (Risk)", f"-{results['max_drawdown_pct']:.2f}%", "Lower is safer")
                k5.metric("⚖️ Reward / Risk", f"{results['risk_reward_ratio']:.2f}x", "Earns ₹ vs ₹1 Risk")
                
                st.markdown("### 📈 Visual Chart — Look Where It Bought & Sold")
                signals_df = results["signals_df"]
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=(f"{display_symbol_name(selected_sym)} Price Chart (▲ Green = BUY, ▼ Red = SELL)", "Trading Volume"),
                    row_heights=[0.75, 0.25]
                )
                
                fig.add_trace(go.Candlestick(
                    x=signals_df.index,
                    open=signals_df["Open"],
                    high=signals_df["High"],
                    low=signals_df["Low"],
                    close=signals_df["Close"],
                    name="Stock Price",
                    increasing_line_color="#10b981",
                    decreasing_line_color="#f43f5e"
                ), row=1, col=1)
                
                buy_signals = signals_df[signals_df["Signal"] == 1]
                sell_signals = signals_df[signals_df["Signal"] == -1]
                
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals["Low"] * 0.995,
                        mode="markers",
                        name="🟢 BUY Entry",
                        marker=dict(symbol="triangle-up", size=14, color="#10b981", line=dict(width=1.5, color="#ffffff"))
                    ), row=1, col=1)
                    
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals["High"] * 1.005,
                        mode="markers",
                        name="🔴 SELL / Exit",
                        marker=dict(symbol="triangle-down", size=14, color="#f43f5e", line=dict(width=1.5, color="#ffffff"))
                    ), row=1, col=1)
                    
                if "Volume" in signals_df.columns:
                    colors = ["#10b981" if c >= o else "#f43f5e" for o, c in zip(signals_df["Open"], signals_df["Close"])]
                    fig.add_trace(go.Bar(
                        x=signals_df.index,
                        y=signals_df["Volume"],
                        name="Volume",
                        marker_color=colors
                    ), row=2, col=1)
                    
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111622",
                    plot_bgcolor="#080b11",
                    xaxis_rangeslider_visible=False,
                    height=520,
                    margin=dict(l=30, r=30, t=40, b=30),
                    hovermode="x unified",
                    font=dict(family="Inter, sans-serif", color="#94a3b8"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(17,22,34,0.8)")
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 💵 Money Growth Curve (How Your Capital Grew Over Time)")
                eq_df = results["equity_df"]
                if not eq_df.empty:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=eq_df.index,
                        y=eq_df["Equity"],
                        name="ApexTrade Strategy Capital (₹)",
                        line=dict(color="#10b981", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(16, 185, 129, 0.08)"
                    ))
                    fig_eq.add_trace(go.Scatter(
                        x=eq_df.index,
                        y=eq_df["Benchmark_Equity"],
                        name="Regular Buy & Hold Capital (₹)",
                        line=dict(color="#94a3b8", width=1.5, dash="dot")
                    ))
                    fig_eq.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#111622",
                        plot_bgcolor="#080b11",
                        height=350,
                        margin=dict(l=30, r=30, t=30, b=30),
                        hovermode="x unified",
                        font=dict(family="Inter, sans-serif", color="#94a3b8"),
                        yaxis_title="Account Balance (₹)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(17,22,34,0.8)")
                    )
                    fig_eq.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                    fig_eq.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
                    st.plotly_chart(fig_eq, use_container_width=True)
                    
                st.markdown("### 📜 Every Trade Listed Step-by-Step")
                trades_list = results["trades"]
                if trades_list:
                    trades_df = pd.DataFrame(trades_list)
                    trades_df["Profit / Loss (₹)"] = trades_df["net_pnl"].apply(lambda x: format_currency_inr(x))
                    trades_df["Return %"] = trades_df["pnl_pct"].apply(lambda x: f"{x:+.2f}%")
                    trades_df["Bought At"] = trades_df["entry_price"].apply(lambda x: f"₹{x:.2f}")
                    trades_df["Sold At"] = trades_df["exit_price"].apply(lambda x: f"₹{x:.2f}")
                    
                    display_cols = ["entry_date", "exit_date", "side", "quantity", "Bought At", "Sold At", "Profit / Loss (₹)", "Return %", "exit_reason"]
                    st.dataframe(trades_df[display_cols], use_container_width=True, hide_index=True)
                    
                    csv = pd.DataFrame(trades_list).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Trades List (Excel / CSV)",
                        data=csv,
                        file_name=f"trades_{selected_sym}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No trades were generated with these exact settings over this time period.")
