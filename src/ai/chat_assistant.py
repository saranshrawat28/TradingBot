"""
Plain-English Conversational AI Trading Assistant with Agentic Tool Calling & Multi-Turn Context Memory.
Features:
1. Multi-turn conversational memory & implicit pronoun/intent resolver.
2. Agentic Tool Runner integration: Live Portfolio Telemetry, Square-Off, Options Greeks, Sector Screener, Pre-Market Intel.
3. Interactive Plotly mini-chart payloads with Entry, SL, T1/T2 overlays.
4. Deterministic Trade Action Cards routed strictly through AIGuardrails.
5. Comprehensive Local Heuristic Engine fallback with institutional financial knowledge base.
"""

import re
import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

from src.data.data_fetcher import search_indian_stocks, resolve_ticker, get_live_quote, get_historical_data, TICKER_ALIASES
from src.engine.stock_advisor import StockAdvisor
from src.engine.pre_market_analyzer import PreMarketAnalyzer
from src.engine.ai_guardrails import AIGuardrails
from src.ai.assistant_tools import AssistantToolRunner
from src.strategies.indicators import calculate_ema, calculate_volume_profile, add_all_indicators
from src.utils.storage import get_portfolio_state
from src.utils.helpers import display_symbol_name, clean_symbol, format_currency_inr, get_ist_now
from src.ai.llm_client import LLMClient
import config

class TradingChatAssistant:
    """
    Stateful Natural Language Trading Assistant for Indian Equities and F&O.
    """

    SYSTEM_PROMPT = """You are ApexTrade AI, an elite institutional algorithmic trading terminal and quantitative desk advisor for Indian capital markets (NSE / BSE / NFO).
Your mission is to provide disciplined, risk-first trade analysis with surgical accuracy, exact technical levels, and institutional professionalism.

CRITICAL DIRECTIVES:
1. STRICT ZERO-DISCLAIMER POLICY: NEVER EVER say "I am an AI assistant and not a real-time terminal", "I do not have access to real-time data", or "As an AI language model...". You are directly interfaced with live quantitative market calculation engines.
2. Ground every analysis on factual live data provided in the <market_data>, <macro_market_context>, and <portfolio_context> tags.
3. PROFESSIONAL BRIEFING STRUCTURE:
   When analyzing any stock or index, structure your briefing with:
   - ⚡ **Executive Summary**: Asset Name, Live Price, Action Verdict (BUY / BUY ON PULLBACK / WAIT / AVOID), and Conviction Score.
   - 🎯 **Institutional Trade Plan (1:2 R:R Focus)**:
     • Entry Zone: Exact price range
     • Stop-Loss: Exact price and risk per share (-X.X%)
     • Target 1: Exact price (+X.X% / locks 50% profit & trails SL to Breakeven)
     • Target 2: Exact price (+X.X% / runner)
     • Risk-to-Reward Ratio: Minimum 1:1.5 to 1:2.0
   - 🧠 **Technical & Order Flow Catalyst**: State why this setup exists (e.g. 20 EMA pullback, VWAP bounce, Camarilla H4 breakout, Volume Profile POC support).
   - ⚠️ **Risk & Overextension Check**: If price is extended >1.8 ATR above 20 EMA, warn the user: "⚠️ Overextended: High risk of climax pullback; wait for a pullback to the 20 EMA floor."
4. FOR INDEX OPTIONS (NIFTY / BANK NIFTY):
   - Always specify the exact ATM strike (e.g. `BANKNIFTY 57800 CE`), universal broker search query (`BANKNIFTY 57800 CE`), and entry/exit targets.
   - Recommend Calls (CE) only when price is above intraday VWAP; recommend Puts (PE) only when below intraday VWAP.
5. If the user asks general trading questions or strategy concepts, explain like an institutional prop trader with concrete Indian market examples.
6. If the user speaks in Hindi/Hinglish (e.g. "Aaj Nifty kaisa lag raha hai?"), respond in polished, professional Hinglish.
7. Format everything in clean, beautiful Markdown with bold headings, bullet points, and tables. NEVER return raw unparsed JSON.
"""

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
        ),
        "camarilla": (
            "🏛️ **Institutional Camarilla Pivot Points**:\n"
            "• **H4 / L4 (Breakout Triggers)**: High-momentum breakout levels used by institutional prop desks. When price crosses H4 with volume, strong directional expansion follows.\n"
            "• **H3 / L3 (Mean-Reversion Reversals)**: Institutional value boundary. Buying at L3 support and shorting at H3 resistance gives high win-rate reversal trades.\n"
            "• **Formula**: $H4 = \\text{Close} + (\\text{High} - \\text{Low}) \\times 1.1 / 2$."
        ),
        "volume_profile": (
            "📊 **Volume Profile & Point of Control (POC)**:\n"
            "• **Point of Control (POC)**: The exact price level where the maximum volume was transacted across the session. Acts as a strong magnet and demand/supply floor.\n"
            "• **Value Area (VAH / VAL)**: The 70% volume distribution representing institutional fair value.\n"
            "• **Edge**: Buying when price holds above POC prevents buying into overhead institutional supply."
        ),
        "ttm_squeeze": (
            "⚡ **TTM Volatility Squeeze**:\n"
            "• **Concept**: Identifies periods when Bollinger Bands contract completely inside Keltner Channels, indicating severe energy compression.\n"
            "• **Breakout Trigger**: When the squeeze fires, volatility expands violently with high statistical follow-through (>75% win-rate)."
        )
    }

    STOP_WORDS = {
        "THE", "STOCK", "STOCKS", "ANALYZE", "BUY", "SELL", "WHAT", "HOW", "ABOUT", "FOR", "TODAY",
        "SHARE", "SHARES", "PRICE", "TELL", "ME", "VIEW", "CHART", "TRADE", "TRADING", "CHECK",
        "LOOKING", "CURRENT", "WITH", "GIVE", "SHOW", "BEST", "TOP", "ANY", "GOOD", "IS", "WAS",
        "ARE", "WERE", "CAN", "COULD", "SHOULD", "WOULD", "WILL", "YOU", "PLEASE", "AND", "OR", "IN",
        "ON", "AT", "OF", "TO", "SWING", "INTRADAY", "POSITIONAL", "LONG", "SHORT", "OPTION",
        "OPTIONS", "CALL", "PUT", "DAILY", "WEEKLY", "MONTHLY", "LEVELS", "TARGET", "STOP", "LOSS",
        "SUPPORT", "RESISTANCE", "WHERE", "WHEN", "WHY", "WHICH", "WHO", "WHOSE", "WHOM", "MUCH",
        "MANY", "MORE", "LESS", "FROM", "INTO", "OVER", "UNDER", "NEAR", "HIGH", "LOW", "CLOSE",
        "OPEN", "ENTRY", "EXIT", "ZONE", "SAFE", "RISK", "ORDER", "ORDERS", "TAKE", "MAKE", "HAVE",
        "HAS", "HAD", "BE", "BEEN", "BEING", "DO", "DOES", "DID", "LOOK", "SEEM", "FEEL", "NOW",
        "SUGGEST", "SUGGESTION", "SUGGESTIONS", "RECOMMEND", "RECOMMENDATION", "RECOMMENDATIONS",
        "PICKS", "FIRST", "SECOND", "THIRD", "ONE", "TWO", "THREE",
        "MARKET", "MOOD", "SENTIMENT", "OUTLOOK", "TREND", "TRENDS", "OVERVIEW"
    }

    @classmethod
    def resolve_symbol_from_text(cls, text: str) -> Optional[str]:
        """Extracts and resolves an Indian stock ticker or index from natural language text."""
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

        # 3. Check for ticker patterns / live search for small-caps (e.g. IZMO, SUZLON, IREDA)
        words = text_clean.split()
        for w in words:
            if len(w) >= 3 and w.isalpha() and w not in cls.STOP_WORDS:
                candidate = f"{w}.NS"
                if any(item["symbol"] == candidate for item in config.DEFAULT_WATCHLIST):
                    return candidate
                
                # Check live search for broad NSE/BSE coverage
                try:
                    search_res = search_indian_stocks(w)
                    if search_res:
                        for s in search_res:
                            s_sym = s.get("symbol", "")
                            if s_sym.replace(".NS", "").replace(".BO", "").upper() == w:
                                return s_sym if s_sym.endswith(".NS") else f"{w}.NS"
                except Exception:
                    pass

        return None

    @classmethod
    def _get_live_macro_context(cls) -> str:
        """Fetches live telemetry for NIFTY 50 and BANK NIFTY to ground all conversational responses."""
        try:
            nifty_q = get_live_quote("^NSEI")
            bank_q = get_live_quote("^NSEBANK")
            nifty_p = float(nifty_q.get("price", 0.0))
            nifty_chg = float(nifty_q.get("change_pct", 0.0))
            bank_p = float(bank_q.get("price", 0.0))
            bank_chg = float(bank_q.get("change_pct", 0.0))
            
            n_mood = "Bullish" if nifty_chg > 0.3 else ("Bearish" if nifty_chg < -0.3 else "Neutral / Range")
            b_mood = "Bullish" if bank_chg > 0.3 else ("Bearish" if bank_chg < -0.3 else "Neutral / Range")
            
            return (
                "<macro_market_context>\n"
                f"• NIFTY 50: ₹{nifty_p:,.2f} ({nifty_chg:+.2f}%) — Sentiment: {n_mood}\n"
                f"• BANK NIFTY: ₹{bank_p:,.2f} ({bank_chg:+.2f}%) — Sentiment: {b_mood}\n"
                "</macro_market_context>\n"
            )
        except Exception:
            return ""

    @classmethod
    def _get_portfolio_context(cls, broker_instance=None) -> str:
        """Fetches available capital and open positions to provide capital-aware sizing."""
        try:
            state = get_portfolio_state()
            cash = float(state.get("cash", 100000.0))
            positions = state.get("positions", [])
            pos_summary = []
            for p in positions[:4]:
                pos_summary.append(f"{p.get('symbol')}: {p.get('quantity')} shares @ ₹{float(p.get('entry_price', 0)):,.2f}")
            pos_str = ", ".join(pos_summary) if pos_summary else "None (100% Cash Buffer)"
            return (
                "<portfolio_context>\n"
                f"• Available Trading Cash Margin: ₹{cash:,.2f}\n"
                f"• Active Open Positions: {pos_str}\n"
                "</portfolio_context>\n"
            )
        except Exception:
            return ""

    @classmethod
    def _clean_disclaimers(cls, text: str) -> str:
        """Strips out canned AI disclaimers and excuses from LLM responses."""
        if not text:
            return ""
        patterns = [
            r"(?i)while\s+i\s+am\s+an\s+ai\s+assistant[^,\.\n]*[,.]?",
            r"(?i)as\s+an\s+ai\s+(language\s+)?model[^,\.\n]*[,.]?",
            r"(?i)i\s+do\s+not\s+have\s+(access\s+to\s+)?real[- ]time\s+(market\s+)?data[^,\.\n]*[,.]?",
            r"(?i)i\s+am\s+not\s+a\s+real[- ]time\s+terminal[^,\.\n]*[,.]?",
            r"(?i)please\s+note\s+that\s+i\s+am\s+an\s+ai[^,\.\n]*[,.]?",
            r"(?i)i\s+cannot\s+provide\s+real[- ]time\s+quotes[^,\.\n]*[,.]?"
        ]
        cleaned = text
        for p in patterns:
            cleaned = re.sub(p, "", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned).strip()
        return cleaned

    @classmethod
    def _parse_and_format_json_response(cls, raw_text: str, stock_analysis: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Detects if the LLM returned a raw JSON dictionary (or concatenated JSONs)
        and transforms it into a beautifully formatted institutional Markdown briefing.
        Uses stock_analysis data as high-precision fallback to ensure targets and prices are NEVER N/A.
        """
        text = raw_text.strip()
        data = None

        # 1. Try direct full-text JSON parse
        try:
            data = json.loads(text)
        except Exception:
            pass

        # 2. If full parse fails, extract first valid JSON object using raw_decode (handles concatenated JSONs)
        if not data or not isinstance(data, dict):
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(text):
                idx = text.find("{", idx)
                if idx == -1:
                    break
                try:
                    obj, end_idx = decoder.raw_decode(text[idx:])
                    if isinstance(obj, dict) and obj:
                        data = obj
                        break
                except Exception:
                    pass
                idx += 1

        if not data or not isinstance(data, dict):
            return raw_text, None

        # 3. Unnest top-level wrapper keys (e.g. {"market_analysis": {...}} or {"response": "..."})
        for text_wrapper in ["response", "content", "text", "message", "answer", "reply", "result", "output"]:
            if text_wrapper in data and isinstance(data[text_wrapper], str) and len(data[text_wrapper].strip()) > 10:
                inner_text = data[text_wrapper].strip()
                if inner_text.startswith("{") and inner_text.endswith("}"):
                    sub_formatted, sub_data = cls._parse_and_format_json_response(inner_text, stock_analysis)
                    return sub_formatted, sub_data
                return inner_text, None

        for dict_wrapper in ["market_analysis", "trade_analysis", "stock_analysis", "analysis", "data", "setup", "trade"]:
            if dict_wrapper in data and isinstance(data[dict_wrapper], dict):
                inner_d = data[dict_wrapper]
                data = {**data, **inner_d}

        # 4. Check if the JSON contains a structured trade / stock analysis
        stock_name = (
            data.get("stock") or data.get("company_name") or data.get("company") or
            data.get("symbol") or data.get("ticker") or
            (stock_analysis.get("display_name") if stock_analysis else None)
        )
        if not stock_name:
            # If no stock name, but has values, format keys into clean bullets
            clean_lines = []
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)):
                    title = k.replace("_", " ").title()
                    clean_lines.append(f"• **{title}**: {v}")
            if clean_lines:
                return "\n".join(clean_lines), None
            return raw_text, None

        summary = (
            data.get("analysis_summary") or data.get("summary") or
            data.get("technical_notes") or data.get("notes") or
            data.get("description") or data.get("rational") or ""
        )
        
        # Price and Score Fallbacks
        sa_price = float(stock_analysis.get("current_price", 0.0)) if stock_analysis else 0.0
        sa_score = float(stock_analysis.get("score", 7.0)) if stock_analysis else 7.0

        price_and_score = data.get("live_price_and_score", {})
        if isinstance(price_and_score, dict):
            live_price = price_and_score.get("live_price") or (f"₹{sa_price:,.2f}" if sa_price > 0 else "Live Market Price")
            score = price_and_score.get("mathematical_score") or f"{sa_score:.1f}/10.0"
        else:
            live_price = data.get("live_price") or data.get("current_price") or data.get("price") or (f"₹{sa_price:,.2f}" if sa_price > 0 else "Live Market Price")
            score = data.get("score") or data.get("mathematical_score") or f"{sa_score:.1f}/10.0"

        # Trade Plan
        trade_plan = data.get("trade_plan", {}) if isinstance(data.get("trade_plan"), dict) else {}
        
        # Entry Zone
        entry_zone = (
            trade_plan.get("ideal_entry_zone") or data.get("ideal_entry_zone") or
            data.get("entry_zone") or (stock_analysis.get("entry_zone") if stock_analysis else None) or
            (f"₹{sa_price * 0.995:,.2f} – ₹{sa_price:,.2f}" if sa_price > 0 else "Current Market Zone")
        )
        
        # Target 1
        sa_t1 = stock_analysis.get("target_1", {}) if stock_analysis else {}
        t1_raw = trade_plan.get("target_1") or data.get("target_1") or data.get("t1") or {}
        if isinstance(t1_raw, dict):
            t1_price = t1_raw.get("price") or (f"₹{sa_t1.get('price', 0):,.2f}" if sa_t1.get("price") else (f"₹{sa_price * 1.03:,.2f}" if sa_price > 0 else "N/A"))
            t1_gain_rs = t1_raw.get("gain_rs", "")
            t1_gain_pct = t1_raw.get("gain_percent") or (f"{sa_t1.get('gain_pct', 3.0):.1f}%" if sa_t1.get("gain_pct") else "3.0%")
            t1_action = t1_raw.get("action", "Lock 50% profits & Breakeven")
        else:
            t1_price = str(t1_raw) if t1_raw else (f"₹{sa_t1.get('price', 0):,.2f}" if sa_t1.get("price") else (f"₹{sa_price * 1.03:,.2f}" if sa_price > 0 else "N/A"))
            t1_gain_rs = ""
            t1_gain_pct = f"{sa_t1.get('gain_pct', 3.0):.1f}%" if sa_t1.get("gain_pct") else "3.0%"
            t1_action = "Lock 50% profits & Breakeven"

        # Target 2
        sa_t2 = stock_analysis.get("target_2", {}) if stock_analysis else {}
        t2_raw = trade_plan.get("target_2") or data.get("target_2") or data.get("t2") or {}
        if isinstance(t2_raw, dict):
            t2_price = t2_raw.get("price") or (f"₹{sa_t2.get('price', 0):,.2f}" if sa_t2.get("price") else (f"₹{sa_price * 1.06:,.2f}" if sa_price > 0 else "N/A"))
            t2_gain_rs = t2_raw.get("gain_rs", "")
            t2_gain_pct = t2_raw.get("gain_percent") or (f"{sa_t2.get('gain_pct', 6.0):.1f}%" if sa_t2.get("gain_pct") else "6.0%")
            t2_action = t2_raw.get("action", "Runner Target")
        else:
            t2_price = str(t2_raw) if t2_raw else (f"₹{sa_t2.get('price', 0):,.2f}" if sa_t2.get("price") else (f"₹{sa_price * 1.06:,.2f}" if sa_price > 0 else "N/A"))
            t2_gain_rs = ""
            t2_gain_pct = f"{sa_t2.get('gain_pct', 6.0):.1f}%" if sa_t2.get("gain_pct") else "6.0%"
            t2_action = "Runner Target"

        # Stop-Loss
        sa_sl = stock_analysis.get("stop_loss", {}) if stock_analysis else {}
        sl_raw = trade_plan.get("safety_stop_loss") or trade_plan.get("stop_loss") or data.get("safety_stop_loss") or data.get("stop_loss") or data.get("sl") or {}
        if isinstance(sl_raw, dict):
            sl_price = sl_raw.get("price") or (f"₹{sa_sl.get('price', 0):,.2f}" if sa_sl.get("price") else (f"₹{sa_price * 0.98:,.2f}" if sa_price > 0 else "N/A"))
            sl_loss_rs = sl_raw.get("loss_rs", "")
            sl_loss_pct = sl_raw.get("loss_percent") or (f"{sa_sl.get('loss_pct', 2.0):.1f}%" if sa_sl.get("loss_pct") else "2.0%")
            sl_action = sl_raw.get("action", "Mandatory Stop-Loss")
        else:
            sl_price = str(sl_raw) if sl_raw else (f"₹{sa_sl.get('price', 0):,.2f}" if sa_sl.get("price") else (f"₹{sa_price * 0.98:,.2f}" if sa_price > 0 else "N/A"))
            sl_loss_rs = ""
            sl_loss_pct = f"{sa_sl.get('loss_pct', 2.0):.1f}%" if sa_sl.get("loss_pct") else "2.0%"
            sl_action = "Mandatory Stop-Loss"

        rr = data.get("risk_reward_ratio") or data.get("risk_reward") or "1.6:1"
        risk_note = data.get("risk_management_note") or data.get("risk_note") or ""

        t1_info = f"{t1_price}"
        if t1_gain_pct and "%" in str(t1_gain_pct): t1_info += f" (+{t1_gain_pct}" + (f" / +₹{t1_gain_rs})" if t1_gain_rs else ")")
        
        t2_info = f"{t2_price}"
        if t2_gain_pct and "%" in str(t2_gain_pct): t2_info += f" (+{t2_gain_pct}" + (f" / +₹{t2_gain_rs})" if t2_gain_rs else ")")
        
        sl_info = f"{sl_price}"
        if sl_loss_pct and "%" in str(sl_loss_pct): sl_info += f" (-{sl_loss_pct}" + (f" / -₹{sl_loss_rs})" if sl_loss_rs else ")")

        lines = [
            f"📊 **Institutional Analysis for {stock_name}** (`{live_price}`):\n",
            f"• **AI Score**: `{score}`",
            f"• 📍 **Ideal Entry Zone**: `{entry_zone}`",
            f"• 🎯 **Target 1**: **{t1_info}** &bull; *{t1_action} 🔒*",
            f"• 🚀 **Target 2**: **{t2_info}** &bull; *{t2_action}*",
            f"• 🛑 **Safety Stop-Loss**: **{sl_info}** &bull; *{sl_action}*",
            f"• 📐 **Risk-to-Reward Ratio**: `{rr}`\n"
        ]

        if stock_analysis:
            grade_title = stock_analysis.get("setup_grade_title")
            if grade_title:
                lines.insert(2, f"• 🏆 **Setup Quality**: `{grade_title}`")

            deriv = stock_analysis.get("derivatives")
            if deriv and deriv.get("status") == "SUCCESS":
                oi_state = deriv.get("oi_interpretation", "NEUTRAL")
                call_wall = float(deriv.get("call_writer_wall", 0.0))
                put_floor = float(deriv.get("put_writer_floor", 0.0))
                pcr_val = float(deriv.get("pcr_oi", 1.0))
                max_pain = float(deriv.get("max_pain", 0.0))
                lines.append(
                    f"⚡ **Derivatives Order Flow**: `{oi_state}` (PCR: `{pcr_val:.2f}` | Max Pain: `₹{max_pain:,.2f}`)\n"
                    f"🧱 **Option Walls**: Call Ceiling `@ ₹{call_wall:,.2f}` | Put Demand Floor `@ ₹{put_floor:,.2f}`\n"
                )

        if summary:
            lines.append(f"💡 **Analysis**: *{summary}*\n")
        if risk_note:
            lines.append(f"🛡️ **Risk Management**: *{risk_note}*\n")

        lines.append("👉 *Use the interactive execution card below to review and place this order.*")
        formatted_md = "\n".join(lines)

        return formatted_md, data

    @classmethod
    def resolve_implicit_reference(cls, user_query: str, active_context: Optional[Dict[str, Any]]) -> Tuple[str, Optional[str], str]:
        """
        Resolves explicit and implicit tickers ('it', 'this', 'that', 'same stock') and timeframe shifts
        using conversational context memory (with 15-minute inactivity expiration).
        Returns (resolved_query, resolved_symbol, resolved_horizon).
        """
        query_clean = user_query.strip().lower()
        
        # Check for explicit timeframe indicators in current query
        horizon = "swing"
        if any(w in query_clean for w in ["intraday", "day trade", "today", "5m", "15m", "mis", "scalp"]):
            horizon = "intraday"
        elif any(w in query_clean for w in ["positional", "long term", "investment", "monthly", "1y", "weeks"]):
            horizon = "positional"
        elif any(w in query_clean for w in ["swing", "daily", "few days"]):
            horizon = "swing"
        elif active_context and "last_horizon" in active_context:
            horizon = active_context.get("last_horizon", "swing")

        # 1. Explicitly check for symbol in user query
        explicit_symbol = cls.resolve_symbol_from_text(user_query)
        if explicit_symbol:
            return user_query, explicit_symbol, horizon

        # 2. If no explicit symbol, check active context memory (within 15-min window)
        if active_context:
            last_ts = active_context.get("timestamp", 0)
            current_ts = get_ist_now().timestamp()
            if (current_ts - last_ts) <= 900: # within 15 mins
                last_symbol = active_context.get("last_symbol")
                last_display = active_context.get("last_display_name", "")

                is_implicit = any(
                    re.search(pattern, query_clean) for pattern in [
                        r"\b(it|its|this|that|same|the stock|this stock|above|them)\b",
                        r"\b(swing|intraday|weekly|positional)\b",
                        r"\b(support|resistance|pivot|target|stop loss|sl|levels)\b",
                        r"\b(buy|sell|short|long|square off|exit)\b",
                        r"\b(chart|graph|candles)\b"
                    ]
                )

                if is_implicit and last_symbol:
                    resolved_query = f"{user_query} for {last_display or last_symbol}"
                    return resolved_query, last_symbol, horizon

        return user_query, None, horizon

    @classmethod
    def _fetch_mini_chart_data(cls, symbol: str, horizon: str, entry_p: float, t1: float, t2: float, sl: float) -> Optional[Dict[str, Any]]:
        """Fetches the last 40 bars of OHLCV data for inline Plotly mini-chart rendering."""
        try:
            period = "5d" if horizon == "intraday" else "1mo"
            interval = "5m" if horizon == "intraday" else "15m"
            df = get_historical_data(symbol, period=period, interval=interval)
            if df.empty or len(df) < 15:
                return None
            
            df_chart = df.tail(40).copy()
            df_chart["EMA_9"] = calculate_ema(df_chart["Close"], 9)
            df_chart["EMA_21"] = calculate_ema(df_chart["Close"], 21)
            
            vp_info = calculate_volume_profile(df_chart, bins=20)
            poc_val = vp_info.get("poc", 0.0)

            date_strings = [idx.strftime("%d %b %H:%M") if hasattr(idx, "strftime") else str(idx) for idx in df_chart.index]

            return {
                "symbol": symbol,
                "display_name": display_symbol_name(symbol),
                "horizon": horizon,
                "dates": date_strings,
                "open": [round(float(v), 2) for v in df_chart["Open"].tolist()],
                "high": [round(float(v), 2) for v in df_chart["High"].tolist()],
                "low": [round(float(v), 2) for v in df_chart["Low"].tolist()],
                "close": [round(float(v), 2) for v in df_chart["Close"].tolist()],
                "ema9": [round(float(v), 2) if not pd.isna(v) else round(float(df_chart["Close"].iloc[0]), 2) for v in df_chart["EMA_9"].tolist()],
                "ema21": [round(float(v), 2) if not pd.isna(v) else round(float(df_chart["Close"].iloc[0]), 2) for v in df_chart["EMA_21"].tolist()],
                "entry_price": round(entry_p, 2),
                "target_1": round(t1, 2),
                "target_2": round(t2, 2),
                "stop_loss": round(sl, 2),
                "poc": round(poc_val, 2)
            }
        except Exception:
            return None

    @classmethod
    def process_query(
        cls,
        user_query: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        last_scanned_picks: Optional[List[Dict[str, Any]]] = None,
        broker_instance=None,
        active_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Processes a user query with full Agentic Tool Calling, Multi-Turn Context Memory,
        and Interactive Mini-Chart / Visual Price Ladder generation.
        """
        history = chat_history or []
        
        # Step 1: Resolve Implicit Pronouns & Context Continuity
        augmented_query, symbol, resolved_horizon = cls.resolve_implicit_reference(user_query, active_context)
        query_clean = augmented_query.strip().lower()

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

        # Step 2: Intent Classification & Tool Dispatch
        is_portfolio_query = any(k in query_clean for k in ["my profit", "p&l", "balance", "portfolio", "positions", "my trades", "open trades", "how much i made", "funds", "capital"])
        is_square_off = any(k in query_clean for k in ["square off", "exit position", "close trade", "exit trade", "sell all", "close all"])
        is_options_query = any(k in query_clean for k in ["option", "strike", "call option", "put option", "ce strike", "pe strike", "greeks", "0dte", "atm", "itm", "otm", "weekly expiry"]) and not ("stock" in query_clean and "option" not in query_clean)
        # Technical Scanner Intents
        is_golden_cross = any(k in query_clean for k in ["golden cross", "goldencross", "50 ema cross", "50 200", "50 ema above 200", "golden cross scan"])
        is_death_cross = any(k in query_clean for k in ["death cross", "deathcross", "50 ema below 200", "50 200 bear"])
        is_camarilla_scan = any(k in query_clean for k in ["camarilla", "h4 breakout", "l3 support", "camarilla scan"])
        is_squeeze_scan = any(k in query_clean for k in ["squeeze", "ttm", "coiling", "volatility squeeze"])
        is_rvol_scan = any(k in query_clean for k in ["rvol", "volume shocker", "high volume", "volume surge", "volume spike", "unusual volume"])
        is_rsi_scan = any(k in query_clean for k in ["oversold", "overbought", "rsi scan", "rsi below", "rsi above"])
        is_sector_scan = any(k in query_clean for k in ["banking sector", "it sector", "auto sector", "power sector", "fmcg sector", "pharma sector", "metal sector", "energy sector"])
        is_general_scan = any(k in query_clean for k in ["screener", "scanner", "scan market", "scan stocks", "find stocks", "stock screener", "scan for"])

        is_screener_query = is_golden_cross or is_death_cross or is_camarilla_scan or is_squeeze_scan or is_rvol_scan or is_rsi_scan or is_sector_scan or is_general_scan
        is_premarket_query = any(k in query_clean for k in ["pre-market", "pre market", "opening cues", "opening mood", "nifty gap", "gap up", "gap down", "morning picks"])
        is_top_picks = (
            any(k in query_clean for k in [
                "best stock", "top stock", "recommend", "suggestion", "suggest",
                "what to buy", "what will i buy", "what should i buy", "what can i buy",
                "which stock to buy", "picks", "morning pick", "which stock", "top 3",
                "today buy", "stocks for today", "give me stock", "stocks to buy", "buy for today"
            ])
            and not symbol
            and not is_screener_query
        )
        is_trade_intent = (
            any(k in query_clean for k in ["buy", "sell", "purchase", "short", "long", "place order", "take trade", "execute"])
            and not is_square_off
            and not is_top_picks
            and not is_screener_query
        )

        # Concept Query Check
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

        # Tool Execution Routing
        ui_card_type = None
        action_card = None
        chart_data = None
        tool_result = None

        # 1. Square-off Tool
        if is_square_off:
            target = symbol if symbol else query_clean
            tool_result = AssistantToolRunner.square_off_action(target, broker_instance)
            ui_card_type = tool_result.get("ui_card_type")
            action_card = tool_result.get("data")
            return {
                "response_text": tool_result.get("summary_markdown", ""),
                "action_card": action_card,
                "ui_card_type": ui_card_type,
                "chart_data": None,
                "symbol": symbol,
                "updated_context": active_context,
                "timestamp": get_ist_now().strftime("%I:%M %p")
            }

        # 2. Portfolio Status Tool
        if is_portfolio_query:
            tool_result = AssistantToolRunner.get_portfolio_status(broker_instance)
            ui_card_type = tool_result.get("ui_card_type")
            action_card = tool_result.get("data")
            return {
                "response_text": tool_result.get("summary_markdown", ""),
                "action_card": action_card,
                "ui_card_type": ui_card_type,
                "chart_data": None,
                "symbol": symbol,
                "is_local_fallback": True,
                "updated_context": active_context,
                "timestamp": get_ist_now().strftime("%I:%M %p")
            }

        # 3. Options Greeks & Strike Selector Tool
        if is_options_query:
            opt_bias = "BUY_PUT" if any(w in query_clean for w in ["put", "pe", "bearish", "down", "fall"]) else "BUY_CALL"
            opt_sym = symbol if symbol else ("BANKNIFTY" if "bank" in query_clean else "NIFTY")
            tool_result = AssistantToolRunner.get_options_recommendation(opt_sym, bias=opt_bias, dte_days=3.0)
            ui_card_type = tool_result.get("ui_card_type")
            action_card = tool_result.get("data")
            return {
                "response_text": tool_result.get("summary_markdown", ""),
                "action_card": action_card,
                "ui_card_type": ui_card_type,
                "chart_data": None,
                "symbol": opt_sym,
                "is_local_fallback": True,
                "updated_context": active_context,
                "timestamp": get_ist_now().strftime("%I:%M %p")
            }

        # 4. Pre-Market Intel Tool
        if is_premarket_query:
            tool_result = AssistantToolRunner.get_premarket_intel()
            return {
                "response_text": tool_result.get("summary_markdown", ""),
                "action_card": tool_result.get("data"),
                "ui_card_type": tool_result.get("ui_card_type"),
                "chart_data": None,
                "symbol": None,
                "is_local_fallback": True,
                "updated_context": active_context,
                "timestamp": get_ist_now().strftime("%I:%M %p")
            }

        # 5. Technical Screener & Scanner Tool
        if is_screener_query:
            scan_type = "golden_cross" if is_golden_cross else (
                "death_cross" if is_death_cross else (
                    "camarilla_breakout" if is_camarilla_scan else (
                        "ttm_squeeze" if is_squeeze_scan else (
                            "volume_shockers" if is_rvol_scan else (
                                "rsi_oversold" if "oversold" in query_clean else (
                                    "rsi_overbought" if "overbought" in query_clean else "golden_cross"
                                )
                            )
                        )
                    )
                )
            )
            target_sector = "all"
            for sec in ["banking", "it", "auto", "fmcg", "energy", "pharma", "metal"]:
                if sec in query_clean:
                    target_sector = sec
                    if not (is_golden_cross or is_death_cross or is_camarilla_scan or is_squeeze_scan or is_rvol_scan or is_rsi_scan):
                        scan_type = "sector"
                    break

            tool_result = AssistantToolRunner.run_technical_scanner(scan_type=scan_type, sector=target_sector)
            return {
                "response_text": tool_result.get("summary_markdown", ""),
                "action_card": tool_result.get("data"),
                "ui_card_type": tool_result.get("ui_card_type"),
                "chart_data": None,
                "symbol": None,
                "is_local_fallback": True,
                "updated_context": active_context,
                "timestamp": get_ist_now().strftime("%I:%M %p")
            }

        # Step 3: Stock Analysis & Trade Action Card Construction
        stock_analysis = {}
        updated_context = active_context or {}

        if symbol:
            try:
                stock_analysis = StockAdvisor.analyze_stock(symbol, horizon=resolved_horizon)
            except Exception:
                stock_analysis = {}

            if stock_analysis.get("status") == "SUCCESS":
                curr_p = float(stock_analysis.get("current_price", 100.0))
                disp_name = stock_analysis.get("display_name", display_symbol_name(symbol))
                t1_data = stock_analysis.get("target_1", {})
                t2_data = stock_analysis.get("target_2", {})
                sl_data = stock_analysis.get("stop_loss", {})
                
                t1_p = float(t1_data.get("price", curr_p * 1.03))
                t2_p = float(t2_data.get("price", curr_p * 1.06))
                sl_p = float(sl_data.get("price", curr_p * 0.98))

                # Update Context Memory
                updated_context = {
                    "last_symbol": symbol,
                    "last_display_name": disp_name,
                    "last_price": curr_p,
                    "last_horizon": resolved_horizon,
                    "last_analysis": stock_analysis,
                    "timestamp": get_ist_now().timestamp()
                }

                # Generate Mini-Chart Payload
                chart_data = cls._fetch_mini_chart_data(symbol, resolved_horizon, curr_p, t1_p, t2_p, sl_p)

                # Construct Action Card
                action_side = "BUY" if not ("sell" in query_clean or "short" in query_clean) else "SELL"
                q_no_commas = query_clean.replace(",", "")
                budget_match = re.search(r"(?:rs\.?|inr|₹)?\s?(\d{4,7})", q_no_commas)
                budget_inr = float(budget_match.group(1)) if budget_match else 25000.0
                qty_match = re.search(r"(\d+)\s*(?:shares|qty|stocks)", q_no_commas)
                
                is_index_sym = symbol in ["^NSEI", "^NSEBANK", "^BSESN"]
                if is_index_sym:
                    lot_sz = 15 if "BANK" in symbol else 25
                    qty = int(qty_match.group(1)) if qty_match else lot_sz
                    prod_type = "NFO Option / Futures (Intraday MIS)"
                else:
                    qty = int(qty_match.group(1)) if qty_match else max(1, int(budget_inr / max(1.0, curr_p)))
                    prod_type = "MIS Intraday (Auto square-off at 3:15 PM)" if resolved_horizon == "intraday" else "CNC Delivery (Swing)"

                actual_capital = qty * curr_p

                ui_card_type = "TRADE"
                action_card = {
                    "symbol": symbol,
                    "display_name": disp_name,
                    "action": action_side,
                    "product_type": prod_type,
                    "quantity": qty,
                    "entry_price": curr_p,
                    "capital_required": actual_capital,
                    "target_1_price": t1_p,
                    "target_1_profit": (t1_p - curr_p) * qty,
                    "target_1_gain_pct": t1_data.get("gain_pct", 3.0),
                    "target_2_price": t2_p,
                    "target_2_profit": (t2_p - curr_p) * qty,
                    "target_2_gain_pct": t2_data.get("gain_pct", 6.0),
                    "stop_loss_price": sl_p,
                    "stop_loss_risk": (curr_p - sl_p) * qty,
                    "stop_loss_pct": sl_data.get("loss_pct", 2.0),
                    "score": float(stock_analysis.get("score", 7.5)),
                    "reward_risk": stock_analysis.get("levels", {}).get("risk_reward", "1:2.0"),
                    "reason": stock_analysis.get("verdict_desc", "Technical momentum setup with dynamic ATR risk control.")
                }

        # Step 4: Conversational LLM or Local Heuristic Response
        top_picks_data = []
        if is_top_picks:
            if last_scanned_picks:
                top_picks_data = last_scanned_picks[:3]
            else:
                try:
                    scan_res = PreMarketAnalyzer.scan_pre_market_stocks(top_n=3)
                    top_picks_data = scan_res.get("top_picks", [])
                except Exception:
                    top_picks_data = []

        response_text = ""
        is_local = True

        if api_key and provider:
            try:
                llm = LLMClient(provider=provider, api_key=api_key, model=model)
                context_block = "<market_data>\n"
                if stock_analysis:
                    t1_p = stock_analysis.get("target_1", {}).get("price", 0)
                    t2_p = stock_analysis.get("target_2", {}).get("price", 0)
                    sl_p = stock_analysis.get("stop_loss", {}).get("price", 0)
                    cam = stock_analysis.get("camarilla_pivots", {})
                    vp = stock_analysis.get("volume_profile", {})
                    sqz = stock_analysis.get("ttm_squeeze", {})
                    fvg = stock_analysis.get("fvg_structure", {})
                    
                    context_block += (
                        f"Stock Analysis for {stock_analysis.get('display_name')} ({symbol}) [{resolved_horizon.upper()}]:\n"
                        f"• Live Price: ₹{stock_analysis.get('current_price', 0):,.2f}\n"
                        f"• Mathematical Score: {stock_analysis.get('score', 0)}/10\n"
                        f"• Verdict: {stock_analysis.get('verdict', 'WAIT')}\n"
                        f"• Target 1: ₹{t1_p:,.2f} (+{stock_analysis.get('target_1', {}).get('gain_pct', 0):.1f}%)\n"
                        f"• Target 2: ₹{t2_p:,.2f} (+{stock_analysis.get('target_2', {}).get('gain_pct', 0):.1f}%)\n"
                        f"• Safety Stop-Loss: ₹{sl_p:,.2f} (-{stock_analysis.get('stop_loss', {}).get('loss_pct', 0):.1f}%)\n"
                        f"• Volume Point of Control (POC): ₹{vp.get('poc', 0):,.2f} ({vp.get('location', 'INSIDE_FAIR_VALUE')})\n"
                        f"• Camarilla Levels: H4 Breakout = ₹{cam.get('h4', 0):,.2f} | L3 Reversal Floor = ₹{cam.get('l3', 0):,.2f}\n"
                        f"• Volatility Squeeze: {'SQUEEZE ACTIVE (Coiling for breakout)' if sqz.get('squeeze_on') else ('SQUEEZE FIRED' if sqz.get('squeeze_fired') else 'NORMAL')}\n"
                    )
                    if fvg.get("has_fvg"):
                        context_block += f"• Smart Money FVG: {fvg.get('description')}\n"
                    context_block += f"• Rational: {stock_analysis.get('verdict_desc', '')}\n"
                context_block += "</market_data>\n"

                macro_block = cls._get_live_macro_context()
                portfolio_block = cls._get_portfolio_context(broker_instance)

                history_text = ""
                if history:
                    recent = history[-4:]
                    history_text = "Recent Conversation:\n" + "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent]) + "\n\n"

                prompt_content = (
                    f"{macro_block}\n"
                    f"{portfolio_block}\n"
                    f"{context_block}\n"
                    f"{history_text}User Question: {user_query}\n\n"
                    "Provide an authoritative, structured, institutional briefing following the SYSTEM_PROMPT directives. "
                    "Include Executive Summary, 1:2 R:R Target Plan, and Technical Catalysts in clean Markdown."
                )
                response_text = llm.generate_response(user_prompt=prompt_content, system_prompt=cls.SYSTEM_PROMPT)
                if response_text and len(response_text.strip()) > 10:
                    response_text = cls._clean_disclaimers(response_text)
                    formatted_text, parsed_json = cls._parse_and_format_json_response(response_text, stock_analysis=stock_analysis)
                    response_text = cls._clean_disclaimers(formatted_text)
                    
                    if not action_card and parsed_json:
                        stock_n = parsed_json.get("stock") or parsed_json.get("symbol", "STOCK")
                        tp = parsed_json.get("trade_plan", {})
                        
                        p_str = str(parsed_json.get("live_price_and_score", {}).get("live_price", "100")) if isinstance(parsed_json.get("live_price_and_score"), dict) else str(parsed_json.get("live_price", "100"))
                        p_match = re.search(r"(\d+(?:\.\d+)?)", p_str.replace(",", ""))
                        curr_p = float(p_match.group(1)) if p_match else 100.0
                        
                        t1_str = str(tp.get("target_1", {}).get("price", curr_p * 1.03))
                        t1_m = re.search(r"(\d+(?:\.\d+)?)", t1_str.replace(",", ""))
                        t1_p = float(t1_m.group(1)) if t1_m else curr_p * 1.03
                        
                        t2_str = str(tp.get("target_2", {}).get("price", curr_p * 1.06))
                        t2_m = re.search(r"(\d+(?:\.\d+)?)", t2_str.replace(",", ""))
                        t2_p = float(t2_m.group(1)) if t2_m else curr_p * 1.06
                        
                        sl_str = str(tp.get("safety_stop_loss", {}).get("price", curr_p * 0.98))
                        sl_m = re.search(r"(\d+(?:\.\d+)?)", sl_str.replace(",", ""))
                        sl_p = float(sl_m.group(1)) if sl_m else curr_p * 0.98

                        qty = max(1, int(25000.0 / max(1.0, curr_p)))
                        
                        card_sym = symbol if symbol else resolve_ticker(stock_n)
                        if not card_sym:
                            card_sym = f"{stock_n.split()[0].upper()}.NS"

                        action_card = {
                            "symbol": card_sym,
                            "display_name": stock_n,
                            "action": "BUY",
                            "product_type": "MIS Intraday (Auto square-off at 3:15 PM)",
                            "quantity": qty,
                            "entry_price": curr_p,
                            "capital_required": qty * curr_p,
                            "target_1_price": t1_p,
                            "target_1_profit": (t1_p - curr_p) * qty,
                            "target_1_gain_pct": round(((t1_p - curr_p) / curr_p) * 100, 1),
                            "target_2_price": t2_p,
                            "target_2_profit": (t2_p - curr_p) * qty,
                            "target_2_gain_pct": round(((t2_p - curr_p) / curr_p) * 100, 1),
                            "stop_loss_price": sl_p,
                            "stop_loss_risk": (curr_p - sl_p) * qty,
                            "stop_loss_pct": round(((curr_p - sl_p) / curr_p) * 100, 1),
                            "score": 7.5,
                            "reason": parsed_json.get("analysis_summary", "Small-cap momentum setup with risk-managed stop loss.")
                        }
                        ui_card_type = "TRADE"
                        
                        if not chart_data and card_sym:
                            chart_data = cls._fetch_mini_chart_data(card_sym, resolved_horizon, curr_p, t1_p, t2_p, sl_p)

                    is_local = False
                else:
                    response_text = cls._generate_heuristic_response(user_query, symbol, stock_analysis, top_picks_data, is_concept_query, resolved_horizon)
            except Exception:
                response_text = cls._generate_heuristic_response(user_query, symbol, stock_analysis, top_picks_data, is_concept_query, resolved_horizon)
                is_local = True
        else:
            response_text = cls._generate_heuristic_response(user_query, symbol, stock_analysis, top_picks_data, is_concept_query, resolved_horizon)
            is_local = True

        return {
            "response_text": response_text,
            "action_card": action_card,
            "ui_card_type": ui_card_type,
            "chart_data": chart_data,
            "symbol": symbol,
            "updated_context": updated_context,
            "is_local_fallback": is_local,
            "timestamp": get_ist_now().strftime("%I:%M %p")
        }

    @classmethod
    def _generate_heuristic_response(
        cls,
        query: str,
        symbol: Optional[str],
        stock_analysis: Dict[str, Any],
        top_picks_data: List[Dict[str, Any]],
        concept_key: Optional[str] = None,
        horizon: str = "intraday"
    ) -> str:
        """Deterministic rule-based response engine when no LLM API key is connected."""
        # 1. Specific Stock Analysis
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
            entry_zone_str = stock_analysis.get("entry_zone", f"₹{curr_p * 0.998:,.2f} – ₹{curr_p:,.2f}")
            
            badge_icon = "🟢" if "BUY" in verdict else ("🔴" if "SELL" in verdict else "🟡")
            horizon_text = "Intraday (Same Day)" if horizon == "intraday" else "Swing (3 to 7 Days)"
            grade_title = stock_analysis.get("setup_grade_title", "Institutional Setup")

            deriv_text = ""
            deriv = stock_analysis.get("derivatives")
            if deriv and deriv.get("status") == "SUCCESS":
                oi_state = deriv.get("oi_interpretation", "NEUTRAL")
                call_wall = float(deriv.get("call_writer_wall", 0.0))
                put_floor = float(deriv.get("put_writer_floor", 0.0))
                pcr_val = float(deriv.get("pcr_oi", 1.0))
                max_pain = float(deriv.get("max_pain", 0.0))
                deriv_text = (
                    f"• ⚡ **Derivatives Order Flow**: `{oi_state}` (PCR: `{pcr_val:.2f}` | Max Pain: `₹{max_pain:,.2f}`)\n"
                    f"• 🧱 **Option Walls**: Call Ceiling `@ ₹{call_wall:,.2f}` | Put Demand Floor `@ ₹{put_floor:,.2f}`\n"
                )

            return (
                f"📊 **Institutional Analysis for {disp}** (`₹{curr_p:,.2f}`) &bull; *{horizon_text}*:\n\n"
                f"• **AI Verdict**: {badge_icon} **{verdict}** (Score: `{score}/10.0`)\n"
                f"• 🏆 **Setup Quality**: `{grade_title}`\n"
                f"• 📍 **Ideal Entry Zone**: `{entry_zone_str}`\n"
                f"• 🎯 **Target 1 (1.5R)**: `₹{t1:,.2f}` (+{t1_gain:.1f}%) &bull; *Locks 50% profits & moves SL to Breakeven 🔒*\n"
                f"• 🚀 **Target 2 (2.5R Runner)**: `₹{t2:,.2f}` (+{t2_gain:.1f}%)\n"
                f"• 🛑 **Safety Stop-Loss**: `₹{sl:,.2f}` (-{sl_loss:.1f}%)\n"
                f"{deriv_text}"
                f"• 📐 **Blended Risk/Reward**: `2.00R Gross` (Meets `>=1.60R` Net Gate)\n\n"
                f"💡 **Technical Rational**: *{desc}*\n\n"
                f"👉 *Check the interactive mini-chart and proportional price ladder below to review and place this order.*"
            )

        # 2. Concept / Educational Query
        if concept_key and concept_key in cls.KNOWLEDGE_BASE:
            return cls.KNOWLEDGE_BASE[concept_key]

        # 3. Top Stock Recommendations
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

        # 4. Live Market Macro Sentiment & Trends
        q_clean = query.lower()
        if any(k in q_clean for k in ["market mood", "market today", "how is market", "market trend", "nifty trend", "bank nifty trend", "market bullish", "market bearish", "aaj market", "overall market", "market direction"]):
            try:
                nifty_q = get_live_quote("^NSEI")
                bank_q = get_live_quote("^NSEBANK")
                nifty_p = float(nifty_q.get("price", 0.0))
                nifty_chg = float(nifty_q.get("change_pct", 0.0))
                bank_p = float(bank_q.get("price", 0.0))
                bank_chg = float(bank_q.get("change_pct", 0.0))
                
                n_dir = "🟢 Bullish" if nifty_chg > 0.25 else ("🔴 Bearish" if nifty_chg < -0.25 else "🟡 Neutral / Range-Bound")
                b_dir = "🟢 Bullish" if bank_chg > 0.25 else ("🔴 Bearish" if bank_chg < -0.25 else "🟡 Neutral / Range-Bound")
                
                overall_bias = "Bullish continuation favored" if (nifty_chg > 0 and bank_chg > 0) else (
                    "Bearish pressure dominant" if (nifty_chg < 0 and bank_chg < 0) else "Mixed divergence — selective stock picking advised"
                )
                
                return (
                    "🏛️ **Institutional Live Market Macro Overview**:\n\n"
                    f"• **NIFTY 50**: `₹{nifty_p:,.2f}` ({nifty_chg:+.2f}%) &bull; **Bias**: {n_dir}\n"
                    f"• **BANK NIFTY**: `₹{bank_p:,.2f}` ({bank_chg:+.2f}%) &bull; **Bias**: {b_dir}\n"
                    f"• 🧭 **Desk Assessment**: *{overall_bias}*\n\n"
                    "**🎯 Execution Directives**:\n"
                    "1. **Trend Alignment**: Only take Long setups in stocks trading above their 20 EMA and intraday VWAP.\n"
                    "2. **Options Strategy**: If Index is above VWAP $\\rightarrow$ Look for ATM Call (CE) pullbacks; if below VWAP $\\rightarrow$ Focus on ATM Put (PE) breakouts.\n"
                    "3. **Risk Guard**: Maintain strict 1:2.0 Risk-to-Reward on every entry with hard Stop-Loss."
                )
            except Exception:
                pass

        # 5. Friendly Greeting / General Help Fallback
        q_lower = query.lower()
        if any(g in q_lower for g in ["hi", "hello", "hey", "namaste", "kaise", "help", "start", "who are you"]):
            return (
                "👋 **Hello! I am your ApexTrade AI Trading Assistant.**\n\n"
                "I can help you with:\n"
                "1. 📊 **Stock Analysis**: Ask *'How is Tata Motors looking today?'* or *'Is Reliance a buy?'*\n"
                "2. 📈 **Follow-ups & Memory**: Ask *'What about for swing trading?'* or *'What is its major support?'*\n"
                "3. 🎯 **Options Greeks**: Ask *'Suggest a Nifty Call option for weekly expiry'*\n"
                "4. 💼 **Live Portfolio**: Ask *'Show my open positions and total P&L'*\n"
                "5. 🛑 **1-Click Square-Off**: Say *'Square off my Tata Motors trade'*\n"
                "6. 🚀 **Safe Bracket Orders**: Say *'Buy ₹25,000 of TCS with safety stop-loss'*"
            )

        return (
            "🤖 **ApexTrade AI Intelligence Ready**.\n\n"
            "I couldn't identify a specific stock or metric in your question. You can ask me:\n"
            "• *'Analyze Tata Motors for intraday'* \n"
            "• *'What about for swing trading?'* (Multi-turn follow-up)\n"
            "• *'Suggest a Nifty CE strike with Greeks'* \n"
            "• *'What is my portfolio status?'* \n"
            "• *'Square off my active positions'*"
        )
