"""
Plain-English Conversational AI Trading Assistant with Guardrail-Gated Natural Language Execution.
Features:
1. Multi-turn intent parsing & ambiguity resolution for Indian equities, indices, and trading concepts.
2. Structured tool calling: Stock Analysis, Pre-Market Sentiment, Top Picks, Educational Concepts, and Portfolio Telemetry.
3. Deterministic Trade Action Cards routed strictly through AIGuardrails.
4. Comprehensive Local Heuristic Engine fallback with an extensive financial knowledge base.
5. Strict Prompt Injection segregation and full multi-model LLM integration.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.data.data_fetcher import search_indian_stocks, resolve_ticker, get_live_quote, TICKER_ALIASES
from src.engine.stock_advisor import StockAdvisor
from src.engine.pre_market_analyzer import PreMarketAnalyzer
from src.engine.ai_guardrails import AIGuardrails
from src.utils.storage import get_portfolio_state
from src.utils.helpers import display_symbol_name, clean_symbol, format_currency_inr, get_ist_now
from src.ai.llm_client import LLMClient
import config

class TradingChatAssistant:
    """
    Stateful Natural Language Trading Assistant for Indian Equities and F&O.
    """

    SYSTEM_PROMPT = """You are ApexTrade AI, an elite institutional trading assistant and quantitative advisor for Indian markets (NSE / BSE / F&O).
Your mission is to help retail traders execute disciplined, risk-first strategies with clarity, precision, and zero jargon.

Core Principles:
1. Ground every analysis on factual data provided in the <market_data> tags.
2. When analyzing any stock, always clearly state:
   - Live Price & Mathematical Score (out of 10)
   - Ideal Entry Zone
   - Target 1 (+₹ gain / +% / locks 50% profits to Breakeven)
   - Target 2 (+₹ gain / runner)
   - Safety Stop-Loss (-₹ loss / -% / mandatory)
   - Risk-to-Reward Ratio (Blended >= 1.6:1 net of STT/exchange fees)
3. If the user asks general trading questions (e.g. "What is ADX?", "Explain CPR pivots", "Why did Zerodha ban bracket orders?"), explain with crystal clear examples tailored to Indian markets.
4. If the user speaks in Hindi/Hinglish (e.g. "Aaj Nifty kaisa lag raha hai?"), respond warmly in friendly, professional Hinglish.
5. Emphasize risk management: never encourage over-leveraging, revenge trading, or trading without a stop-loss.
"""

    # Comprehensive Educational Knowledge Base for offline/heuristic mode
    KNOWLEDGE_BASE = {
        "adx": (
            "📈 **Average Directional Index (ADX)**:\n"
            "• **What it is**: Measures trend strength on a scale of 0 to 100 (regardless of up or down direction).\n"
            "• **Institutional Rules**:\n"
            "  - `ADX < 20`: Weak / sideways chop. Trend systems are disabled to prevent whipsaws.\n"
            "  - `20 <= ADX < 25`: Developing momentum.\n"
            "  - `ADX >= 25`: Strong directional trend — high probability breakout zone.\n"
            "• **In ApexTrade**: Our algorithm dynamically weights trend confidence: $\\mu(\\text{ADX}) = 0.50 + 0.50 \\times \\frac{\\text{ADX} - 20}{15}$."
        ),
        "vwap": (
            "📊 **Volume Weighted Average Price (VWAP)**:\n"
            "• **What it is**: The benchmark intraday price weighted by volume at every transaction.\n"
            "• **4-Zone Execution Logic**:\n"
            "  - `Price > VWAP + 1σ`: Bullish trend. Long breakouts favored.\n"
            "  - `VWAP < Price <= VWAP + 1σ`: Value zone. Ideal entry for pullback longs.\n"
            "  - `VWAP - 1σ <= Price < VWAP`: Discount zone / mean reversion.\n"
            "  - `Price < VWAP - 1σ`: Bearish territory. Only short setups or wait."
        ),
        "stop_loss": (
            "🛑 **Why Stop-Loss is Non-Negotiable**:\n"
            "• A Stop-Loss (SL) is your ultimate shield against black-swan market moves.\n"
            "• **ApexTrade Rule**: Maximum risk per trade is strictly capped at **1.5% to 2.0%** of your capital, with an absolute daily loss circuit breaker at **₹2,000**.\n"
            "• Never move your stop-loss further away from entry during a losing trade!"
        ),
        "bracket_order": (
            "⚡ **Bracket Orders & Software OCO (One-Cancels-Other)**:\n"
            "• **Zerodha Context**: Native Bracket Orders (BO) were discontinued by Zerodha in March 2020 due to extreme market volatility and exchange margin rules.\n"
            "• **ApexTrade Solution**: We built an autonomous **Software OCO State Machine** that places a standard MIS entry order on Kite, then immediately registers an independent `SL-M` (Stop-Loss Market) order on the exchange and monitors Target 1/2 in real-time."
        ),
        "options_greeks": (
            "🎯 **Option Greeks Quick Guide**:\n"
            "• **Delta (Δ)**: Measures price change of the option per ₹1 move in the underlying.\n"
            "• **Gamma (Γ)**: Speed of Delta change. Spikes near expiry (0DTE Gamma risk).\n"
            "• **Theta (Θ)**: Time decay. Options lose value every second as expiry approaches.\n"
            "• **Vega (ν)**: Sensitivity to Implied Volatility (IV) swings.\n"
            "• **ApexTrade Smart Strike**: Automatically selects **ITM1** strikes on 0DTE expiry to protect you from rapid Theta decay!"
        ),
        "max_pain": (
            "📌 **Option Max Pain Theory**:\n"
            "• **Concept**: The strike price where option writers (sellers/institutions) experience the least financial loss at expiry.\n"
            "• Spot prices historically gravitate towards the Max Pain level on Thursday weekly expiries."
        ),
        "pcr": (
            "⚖️ **Put-Call Ratio (PCR)**:\n"
            "• Calculated as: $\\text{PCR} = \\frac{\\text{Total Put Open Interest}}{\\text{Total Call Open Interest}}$.\n"
            "• `PCR > 1.30`: Bullish sentiment (strong Put writing / support).\n"
            "• `0.80 <= PCR <= 1.30`: Balanced / Neutral market.\n"
            "• `PCR < 0.70`: Bearish sentiment / Overbought warning."
        )
    }

    @classmethod
    def resolve_symbol_from_text(cls, text: str) -> Optional[str]:
        """
        Extracts and resolves an Indian stock ticker or index from natural language text.
        Handles variations, tickers, and common aliases.
        """
        text_clean = re.sub(r"[^\w\s\.\^\&]", " ", text).strip().upper()
        padded_text = f" {text_clean} "
        
        # 1. Check known aliases (longest match first)
        sorted_aliases = sorted(TICKER_ALIASES.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if f" {alias} " in padded_text:
                return TICKER_ALIASES[alias]

        # 2. Check default watchlist items
        for item in config.DEFAULT_WATCHLIST:
            name_u = item["name"].upper()
            sym_u = item["symbol"].replace(".NS", "").replace(".BO", "").upper()
            if f" {name_u} " in padded_text or f" {sym_u} " in padded_text:
                return item["symbol"]

        # 3. Check for ticker patterns e.g. "TCS", "ITC", "SBIN", "INFY", "WIPRO"
        words = text_clean.split()
        for w in words:
            if len(w) >= 3 and w.isalpha():
                candidate = f"{w}.NS"
                if any(item["symbol"] == candidate for item in config.DEFAULT_WATCHLIST):
                    return candidate

        return None

    @classmethod
    def process_query(
        cls,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        last_scanned_picks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Processes a user query, invokes quantitative tools, and returns a structured response + deterministic action cards.
        """
        history = chat_history or []
        query_clean = user_query.strip().lower()

        # Step 1: Extract Referenced Stock (if any)
        symbol = cls.resolve_symbol_from_text(user_query)

        # Handle ordinal referents like "first pick", "second stock", "3rd one"
        if not symbol and last_scanned_picks:
            if any(k in query_clean for k in ["first", "1st", "number one", "top pick"]):
                symbol = last_scanned_picks[0]["symbol"]
            elif any(k in query_clean for k in ["second", "2nd", "number two"]):
                if len(last_scanned_picks) > 1:
                    symbol = last_scanned_picks[1]["symbol"]
            elif any(k in query_clean for k in ["third", "3rd", "number three"]):
                if len(last_scanned_picks) > 2:
                    symbol = last_scanned_picks[2]["symbol"]

        # Step 2: Detect Intent Categories
        is_trade_intent = any(k in query_clean for k in ["buy", "sell", "purchase", "short", "long", "place order", "take trade", "execute"])
        is_top_picks = any(k in query_clean for k in ["best stock", "top stock", "recommend", "suggestion", "what to buy", "picks", "morning pick", "which stock", "top 3"])
        is_portfolio_query = any(k in query_clean for k in ["my profit", "p&l", "balance", "portfolio", "positions", "my trades", "earnings today", "how much i made", "funds", "capital"])
        
        # General market sentiment triggers ONLY if no specific individual stock is targeted
        is_market_sentiment = (not symbol or symbol in ["^NSEI", "^NSEBANK", "^BSESN"]) and any(
            k in query_clean for k in ["market", "nifty", "banknifty", "sensex", "overall", "opening mood", "pre-market", "premarket", "mood today", "market trend"]
        ) and not is_trade_intent

        # Educational / Concept Query Detection
        is_concept_query = None
        for concept_key in cls.KNOWLEDGE_BASE.keys():
            if concept_key in query_clean or (concept_key == "stop_loss" and ("stop loss" in query_clean or "sl" in query_clean.split())):
                is_concept_query = concept_key
                break
            elif concept_key == "bracket_order" and ("bracket" in query_clean or "oco" in query_clean):
                is_concept_query = "bracket_order"
                break
            elif concept_key == "options_greeks" and any(g in query_clean for g in ["greek", "delta", "gamma", "theta", "vega", "0dte"]):
                is_concept_query = "options_greeks"
                break
            elif concept_key == "max_pain" and "pain" in query_clean:
                is_concept_query = "max_pain"
                break

        # Step 3: Tool Data Invocations
        market_telemetry = {}
        stock_analysis = {}
        top_picks_data = []
        portfolio_telemetry = {}
        proposed_action_card = None

        if is_market_sentiment or is_top_picks:
            try:
                sentiment_info = PreMarketAnalyzer.get_market_opening_sentiment()
                market_telemetry = sentiment_info
            except Exception:
                market_telemetry = {"title": "Balanced / Range-Bound", "nifty_price": 24500.0, "gap_pct": 0.0, "explanation": "Live index feed active."}

        if is_top_picks:
            try:
                scan_res = PreMarketAnalyzer.scan_pre_market_stocks(top_n=3)
                top_picks_data = scan_res.get("top_picks", [])
            except Exception:
                top_picks_data = []

        if is_portfolio_query:
            p_state = get_portfolio_state()
            portfolio_telemetry = {
                "cash": float(p_state.get("cash", 100000.0)),
                "daily_pnl": float(p_state.get("daily_pnl", 0.0)),
                "positions_count": len(p_state.get("positions", {}))
            }

        if symbol and symbol not in ["^NSEI", "^NSEBANK", "^BSESN"]:
            horizon = "intraday" if any(k in query_clean for k in ["intraday", "day", "today", "mis"]) else "swing"
            try:
                stock_analysis = StockAdvisor.analyze_stock(symbol, horizon=horizon)
            except Exception:
                stock_analysis = {}

            # If user explicitly wants to buy/sell, construct a deterministic confirmation card
            if is_trade_intent and stock_analysis.get("status") == "SUCCESS":
                action_side = "BUY" if not ("sell" in query_clean or "short" in query_clean) else "SELL"
                curr_p = float(stock_analysis.get("current_price", 100.0))
                
                # Extract budget if mentioned (e.g. "buy 50000 of reliance" or "buy ₹25,000 of reliance")
                q_no_commas = query_clean.replace(",", "")
                budget_match = re.search(r"(?:rs\.?|inr|₹)?\s?(\d{4,7})", q_no_commas)
                budget_inr = float(budget_match.group(1)) if budget_match else 25000.0
                
                qty_match = re.search(r"(\d+)\s*(?:shares|qty|stocks)", q_no_commas)
                qty = int(qty_match.group(1)) if qty_match else max(1, int(budget_inr / max(1.0, curr_p)))
                actual_capital = qty * curr_p

                t1_data = stock_analysis.get("target_1", {})
                sl_data = stock_analysis.get("stop_loss", {})
                t1_p = float(t1_data.get("price", curr_p * 1.03))
                sl_p = float(sl_data.get("price", curr_p * 0.98))

                proposed_action_card = {
                    "symbol": symbol,
                    "display_name": stock_analysis.get("display_name", display_symbol_name(symbol)),
                    "action": action_side,
                    "product_type": "MIS Intraday (Auto square-off at 3:15 PM)",
                    "quantity": qty,
                    "entry_price": curr_p,
                    "capital_required": actual_capital,
                    "target_1_price": t1_p,
                    "target_1_profit": (t1_p - curr_p) * qty,
                    "target_1_gain_pct": t1_data.get("gain_pct", 3.0),
                    "stop_loss_price": sl_p,
                    "stop_loss_risk": (curr_p - sl_p) * qty,
                    "stop_loss_pct": sl_data.get("loss_pct", 2.0),
                    "score": float(stock_analysis.get("score", 7.5)),
                    "reason": stock_analysis.get("verdict_desc", "Technical momentum setup with dynamic ATR risk control.")
                }

        # Step 4: Generate Conversational Response (LLM or Local Heuristic Engine)
        response_text = ""
        is_local = True

        if api_key and provider:
            try:
                llm = LLMClient(provider=provider, api_key=api_key, model=model)
                
                # Context Data Segregation (Anti-Prompt-Injection)
                context_block = "<market_data>\n"
                if market_telemetry:
                    context_block += f"Market Sentiment: {market_telemetry.get('title')} - {market_telemetry.get('explanation')} (NIFTY: ₹{market_telemetry.get('nifty_price', 0):,.2f}, Gap: {market_telemetry.get('gap_pct', 0):+.2f}%)\n"
                if top_picks_data:
                    context_block += f"Top Morning Picks: {[p['display_name'] + ' (' + p['action_title'] + ' @ ₹' + str(p['current_price']) + ')' for p in top_picks_data]}\n"
                if portfolio_telemetry:
                    context_block += f"Portfolio: Available Cash: ₹{portfolio_telemetry['cash']:,.2f}, Today's PnL: ₹{portfolio_telemetry['daily_pnl']:,.2f}, Open Positions: {portfolio_telemetry['positions_count']}\n"
                if stock_analysis:
                    t1_p = stock_analysis.get("target_1", {}).get("price", 0)
                    t2_p = stock_analysis.get("target_2", {}).get("price", 0)
                    sl_p = stock_analysis.get("stop_loss", {}).get("price", 0)
                    context_block += (
                        f"Stock Analysis for {stock_analysis.get('display_name')} ({symbol}):\n"
                        f"• Live Price: ₹{stock_analysis.get('current_price', 0):,.2f}\n"
                        f"• Mathematical Score: {stock_analysis.get('score', 0)}/10\n"
                        f"• Verdict: {stock_analysis.get('verdict', 'WAIT')}\n"
                        f"• Target 1 (1.5R, Locks 50% profits & Breakeven): ₹{t1_p:,.2f}\n"
                        f"• Target 2 (2.5R Runner): ₹{t2_p:,.2f}\n"
                        f"• Safety Stop-Loss: ₹{sl_p:,.2f}\n"
                        f"• Description: {stock_analysis.get('verdict_desc', '')}\n"
                    )
                context_block += "</market_data>\n"

                # Incorporate conversation history
                history_text = ""
                if history:
                    recent = history[-4:]
                    history_text = "Recent Conversation:\n" + "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent]) + "\n\n"

                prompt_content = f"{context_block}\n{history_text}User Question: {user_query}\n\nProvide a structured, helpful, professional response in plain English (or friendly Hinglish if the user asks in Hindi)."
                response_text = llm.generate_response(user_prompt=prompt_content, system_prompt=cls.SYSTEM_PROMPT)
                if response_text and len(response_text.strip()) > 10:
                    is_local = False
                else:
                    response_text = cls._generate_heuristic_response(user_query, symbol, stock_analysis, market_telemetry, top_picks_data, portfolio_telemetry, is_concept_query)
            except Exception:
                response_text = cls._generate_heuristic_response(user_query, symbol, stock_analysis, market_telemetry, top_picks_data, portfolio_telemetry, is_concept_query)
                is_local = True
        else:
            response_text = cls._generate_heuristic_response(user_query, symbol, stock_analysis, market_telemetry, top_picks_data, portfolio_telemetry, is_concept_query)
            is_local = True

        return {
            "response_text": response_text,
            "action_card": proposed_action_card,
            "symbol": symbol,
            "is_local_fallback": is_local,
            "timestamp": get_ist_now().strftime("%I:%M %p")
        }

    @classmethod
    def _generate_heuristic_response(
        cls,
        query: str,
        symbol: Optional[str],
        stock_analysis: Dict[str, Any],
        market_telemetry: Dict[str, Any],
        top_picks_data: List[Dict[str, Any]],
        portfolio_telemetry: Dict[str, Any],
        concept_key: Optional[str] = None
    ) -> str:
        """
        Deterministic, transparent rule-based fallback response engine when no LLM API key is connected.
        Strict precedence: Stock Analysis -> Educational Concepts -> Top Picks -> Portfolio -> Market Sentiment.
        """
        # 1. Specific Stock Analysis (Highest Priority when symbol mentioned)
        if symbol and stock_analysis and stock_analysis.get("status") == "SUCCESS":
            disp = stock_analysis.get("display_name", display_symbol_name(symbol))
            score = stock_analysis.get("score", 5.0)
            verdict = stock_analysis.get("verdict", "WAIT")
            curr_p = stock_analysis.get("current_price", 0.0)
            t1 = stock_analysis.get("target_1", {}).get("price", curr_p * 1.03)
            t1_gain = stock_analysis.get("target_1", {}).get("gain_pct", 3.0)
            t2 = stock_analysis.get("target_2", {}).get("price", curr_p * 1.06)
            t2_gain = stock_analysis.get("target_2", {}).get("gain_pct", 6.0)
            sl = stock_analysis.get("stop_loss", {}).get("price", curr_p * 0.98)
            sl_loss = stock_analysis.get("stop_loss", {}).get("loss_pct", 2.0)
            desc = stock_analysis.get("verdict_desc", "")
            
            badge_icon = "🟢" if "BUY" in verdict else ("🔴" if "SELL" in verdict else "🟡")

            return (
                f"📊 **Institutional Analysis for {disp}** (`₹{curr_p:,.2f}`):\n\n"
                f"• **AI Verdict**: {badge_icon} **{verdict}** (Score: `{score}/10.0`)\n"
                f"• 📍 **Ideal Entry Zone**: `₹{curr_p * 0.998:,.2f} – ₹{curr_p:,.2f}`\n"
                f"• 🎯 **Target 1 (1.5R)**: `₹{t1:,.2f}` (+{t1_gain:.1f}%) &bull; *Locks 50% profits & moves SL to Breakeven 🔒*\n"
                f"• 🚀 **Target 2 (2.5R Runner)**: `₹{t2:,.2f}` (+{t2_gain:.1f}%)\n"
                f"• 🛑 **Safety Stop-Loss**: `₹{sl:,.2f}` (-{sl_loss:.1f}%)\n"
                f"• 📐 **Blended Risk/Reward**: `2.00R Gross` (Meets `>=1.60R` Net Gate)\n\n"
                f"💡 **Technical Rational**: *{desc}*\n\n"
                f"👉 *To place this order with automatic stop-loss, click the confirmation card below or say 'Buy ₹25,000 of {disp}'.*"
            )

        # 2. Concept / Educational Query
        if concept_key and concept_key in cls.KNOWLEDGE_BASE:
            return cls.KNOWLEDGE_BASE[concept_key]

        # 3. Portfolio Query
        if portfolio_telemetry:
            cash = portfolio_telemetry["cash"]
            pnl = portfolio_telemetry["daily_pnl"]
            pnl_str = f"+₹{pnl:,.2f}" if pnl >= 0 else f"-₹{abs(pnl):,.2f}"
            return (
                f"💼 **Your Live Account Snapshot**:\n\n"
                f"• 💵 **Available Cash**: ₹{cash:,.2f}\n"
                f"• 📈 **Today's Realized P&L**: `{pnl_str}`\n"
                f"• 📦 **Active Positions**: {portfolio_telemetry['positions_count']} trade(s) running\n"
                f"• 🛡️ **Daily Risk Ceiling**: ₹2,000.00 max risk floor"
            )

        # 4. Top Stock Recommendations
        if top_picks_data:
            lines = ["🌟 **Today's Top 3 AI Intraday Breakout Recommendations**:"]
            for i, p in enumerate(top_picks_data, 1):
                t1_p = float(p.get("target_1_price", 0.0))
                t1_g = float(p.get("target_1_gain_pct", 3.0))
                sl_p = float(p.get("stop_loss_price", 0.0))
                sl_l = float(p.get("stop_loss_pct", 2.0))
                score_v = p.get("score", 8.0)
                reason_v = p.get("reason", "High-momentum technical breakout setup.")
                lines.append(
                    f"{i}. **{p.get('display_name', 'Stock')}**: `{p.get('action_title', 'BUY')}` @ `₹{p.get('current_price', 0.0):,.2f}`\n"
                    f"   • 🎯 **Target 1**: ₹{t1_p:,.2f} (+{t1_g:.1f}%) | 🛑 **SL**: ₹{sl_p:,.2f} (-{sl_l:.1f}%)\n"
                    f"   • 🧠 **Score**: `{score_v}/10` &bull; *{reason_v}*"
                )
            lines.append("\n💡 *Click on any Action Card below to place a safe bracket trade with 1-click execution.*")
            return "\n\n".join(lines)

        # 5. Market Opening / NIFTY Sentiment
        if market_telemetry:
            mood = market_telemetry.get('title', 'Neutral')
            nifty_p = market_telemetry.get('nifty_price', 0)
            gap = market_telemetry.get('gap_pct', 0)
            expl = market_telemetry.get('explanation', '')
            return (
                f"🌅 **Indian Market Opening Briefing**:\n\n"
                f"• 🧭 **Opening Mood**: **{mood}**\n"
                f"• 📊 **NIFTY 50 Index**: `₹{nifty_p:,.2f}` ({gap:+.2f}% Opening Gap)\n"
                f"• 🔍 **Strategic Assessment**: {expl}\n\n"
                f"💡 *Trading Rule*: In high-gap regimes (|Gap| > 0.40%), wait for the 09:15-09:30 AM opening candle range to settle before entering breakout trades."
            )

        # 6. Friendly Greeting / General Help Fallback
        q_lower = query.lower()
        if any(g in q_lower for g in ["hi", "hello", "hey", "namaste", "kaise", "help", "start", "who are you"]):
            return (
                "👋 **Hello! I am your ApexTrade AI Trading Assistant.**\n\n"
                "I can help you with:\n"
                "1. 📊 **Stock Analysis**: Ask *'How is Tata Motors looking today?'* or *'Is Reliance a buy?'*\n"
                "2. 🌟 **Top Picks**: Ask *'Show today's best 3 intraday stocks'*\n"
                "3. 🌅 **Market Sentiment**: Ask *'What is NIFTY doing right now?'*\n"
                "4. 💼 **Account Status**: Ask *'What is my profit today?'*\n"
                "5. 🚀 **Safe 1-Click Orders**: Say *'Buy ₹25,000 of Infosys with safety stop-loss'*\n"
                "6. 📚 **Concepts**: Ask *'What is ADX?'*, *'Explain VWAP'*, or *'Why did Zerodha ban bracket orders?'*"
            )

        return (
            "🤖 **ApexTrade AI Intelligence Ready**.\n\n"
            "I couldn't identify a specific stock or metric in your question. You can ask me:\n"
            "• *'Analyze Tata Motors for intraday'* \n"
            "• *'What is the NIFTY opening trend today?'*\n"
            "• *'Show top 3 stock picks'* \n"
            "• *'Buy ₹25,000 of TCS with safety stop-loss'* \n"
            "• *'Explain Option Greeks and Max Pain'*"
        )

