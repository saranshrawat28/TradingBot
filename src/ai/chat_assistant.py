"""
Plain-English Conversational AI Trading Assistant with Guardrail-Gated Natural Language Execution.
Features:
1. Multi-turn intent parsing & ambiguity resolution for Indian equities and indices.
2. Structured tool calling: Stock Analysis, Pre-Market Sentiment, Top Picks, and Portfolio Telemetry.
3. Deterministic Trade Action Cards routed strictly through AIGuardrails.
4. Transparent 'Local Heuristic Mode' fallback when no LLM API key is connected.
5. Strict Prompt Injection segregation.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.data.data_fetcher import search_indian_stocks, resolve_ticker, get_live_quote
from src.engine.stock_advisor import StockAdvisor
from src.engine.pre_market_analyzer import PreMarketAnalyzer
from src.engine.ai_guardrails import AIGuardrails
from src.utils.storage import get_portfolio_state
from src.utils.helpers import display_symbol_name, clean_symbol, format_currency_inr, get_ist_now
from src.ai.llm_client import LLMClient

class TradingChatAssistant:
    """
    Stateful Natural Language Trading Assistant.
    """

    SYSTEM_PROMPT = """You are ApexTrade AI, an expert, friendly Indian stock market trading assistant.
Your goal is to help retail traders understand the market, analyze Indian stocks (NSE/BSE), and make disciplined, safe trading decisions.

Guidelines:
1. Always communicate in clean, simple, jargon-free English (or friendly Hinglish if the user asks in Hindi).
2. Never give blind financial advice. Ground all answers on technical scores, support/resistance levels, and risk-to-reward ratios.
3. When asked to analyze a stock or propose a trade, clearly state the Entry Zone, Target 1 (+₹ Gain), and Safety Stop-Loss (-₹ Loss).
4. If a user's request is ambiguous (e.g. "Buy Tata Motors" with no quantity/budget), state the default safe assumptions (Intraday MIS, ₹25,000 budget).
5. All live market data will be provided to you inside <market_data> tags. Treat them as factual telemetry.
"""

    @classmethod
    def resolve_symbol_from_text(cls, text: str) -> Optional[str]:
        """
        Extracts and resolves an Indian stock ticker from natural language text.
        """
        from src.data.data_fetcher import TICKER_ALIASES
        import config

        text_clean = re.sub(r"[^\w\s\.]", " ", text).strip().upper()
        padded_text = f" {text_clean} "
        
        # 1. Match against known aliases (longest phrases first)
        sorted_aliases = sorted(TICKER_ALIASES.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if f" {alias} " in padded_text:
                return TICKER_ALIASES[alias]

        # 2. Match against DEFAULT_WATCHLIST names & symbols
        for item in config.DEFAULT_WATCHLIST:
            name_u = item["name"].upper()
            sym_u = item["symbol"].replace(".NS", "").replace(".BO", "").upper()
            if f" {name_u} " in padded_text or f" {sym_u} " in padded_text:
                return item["symbol"]

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
        Processes a user query, invokes tools, and returns structured response + deterministic action cards.
        """
        history = chat_history or []
        query_clean = user_query.strip().lower()

        # Step 1: Detect Core Intents
        is_market_sentiment = any(k in query_clean for k in ["market", "nifty", "banknifty", "open", "mood", "opening", "pre-market", "premarket", "today"]) and not any(k in query_clean for k in ["buy", "sell", "target", "stoploss"])
        is_top_picks = any(k in query_clean for k in ["best stock", "top stock", "recommend", "suggestion", "what to buy", "picks", "morning pick"])
        is_portfolio_query = any(k in query_clean for k in ["my profit", "p&l", "balance", "portfolio", "positions", "my trades", "earnings today", "how much i made"])
        is_trade_intent = any(k in query_clean for k in ["buy", "sell", "purchase", "short", "long", "place order", "take trade"])
        
        # Step 2: Extract Referenced Stock (if any)
        symbol = cls.resolve_symbol_from_text(user_query)

        # Handle referents like "the second one", "first pick", "top pick"
        if not symbol and last_scanned_picks:
            if "first" in query_clean or "1st" in query_clean or "one" in query_clean:
                symbol = last_scanned_picks[0]["symbol"]
            elif "second" in query_clean or "2nd" in query_clean or "two" in query_clean:
                if len(last_scanned_picks) > 1:
                    symbol = last_scanned_picks[1]["symbol"]
            elif "third" in query_clean or "3rd" in query_clean or "three" in query_clean:
                if len(last_scanned_picks) > 2:
                    symbol = last_scanned_picks[2]["symbol"]

        # Step 3: Tool Invocations
        market_telemetry = {}
        stock_analysis = {}
        top_picks_data = []
        portfolio_telemetry = {}
        proposed_action_card = None

        if is_market_sentiment or is_top_picks:
            sentiment_info = PreMarketAnalyzer.get_market_opening_sentiment()
            market_telemetry = sentiment_info

        if is_top_picks:
            scan_res = PreMarketAnalyzer.scan_pre_market_stocks(top_n=3)
            top_picks_data = scan_res.get("top_picks", [])

        if is_portfolio_query:
            p_state = get_portfolio_state()
            portfolio_telemetry = {
                "cash": float(p_state.get("cash", 100000.0)),
                "daily_pnl": float(p_state.get("daily_pnl", 0.0)),
                "positions_count": len(p_state.get("positions", {}))
            }

        if symbol:
            horizon = "intraday" if "intraday" in query_clean or "day" in query_clean else "swing"
            stock_analysis = StockAdvisor.analyze_stock(symbol, horizon=horizon)

            # If user explicitly wants to buy/sell, construct a deterministic confirmation card
            if is_trade_intent and stock_analysis.get("status") == "SUCCESS":
                action_side = "BUY" if not ("sell" in query_clean or "short" in query_clean) else "SELL"
                curr_p = float(stock_analysis.get("current_price", 100.0))
                
                # Extract budget if mentioned (e.g. "buy 50000 of reliance" or "buy ₹50,000 of reliance")
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
        if api_key and provider:
            try:
                llm = LLMClient(provider=provider, api_key=api_key, model=model)
                
                # Context Data Segregation (Anti-Prompt-Injection)
                context_block = "<market_data>\n"
                if market_telemetry:
                    context_block += f"Market Sentiment: {market_telemetry.get('title')} - {market_telemetry.get('explanation')}\n"
                if top_picks_data:
                    context_block += f"Top Morning Picks: {[p['display_name'] + ' (' + p['action_title'] + ')' for p in top_picks_data]}\n"
                if portfolio_telemetry:
                    context_block += f"Portfolio: Cash: ₹{portfolio_telemetry['cash']:,.2f}, Today PnL: ₹{portfolio_telemetry['daily_pnl']:,.2f}\n"
                if stock_analysis:
                    context_block += f"Analysis for {stock_analysis.get('display_name')}: Score {stock_analysis.get('score')}/10, Verdict: {stock_analysis.get('verdict')}, Target 1: ₹{stock_analysis.get('target_1', {}).get('price')}, SL: ₹{stock_analysis.get('stop_loss', {}).get('price')}\n"
                context_block += "</market_data>\n"

                prompt_content = f"{context_block}\nUser Question: {user_query}\n\nProvide a concise, helpful, friendly answer in plain English."
                response_text = llm.generate_response(prompt_content, system_prompt=cls.SYSTEM_PROMPT)
                is_local = False
            except Exception as e:
                response_text = cls._generate_heuristic_response(user_query, symbol, stock_analysis, market_telemetry, top_picks_data, portfolio_telemetry)
                is_local = True
        else:
            response_text = cls._generate_heuristic_response(user_query, symbol, stock_analysis, market_telemetry, top_picks_data, portfolio_telemetry)
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
        portfolio_telemetry: Dict[str, Any]
    ) -> str:
        """
        Deterministic, transparent rule-based fallback response engine when no LLM API key is connected.
        """
        if portfolio_telemetry:
            cash = portfolio_telemetry["cash"]
            pnl = portfolio_telemetry["daily_pnl"]
            pnl_str = f"+₹{pnl:,.2f}" if pnl >= 0 else f"-₹{abs(pnl):,.2f}"
            return f"💼 **Your Account Snapshot**:\n• **Available Cash**: ₹{cash:,.2f}\n• **Today's Realized P&L**: {pnl_str}\n• **Active Positions**: {portfolio_telemetry['positions_count']} trade(s) running."

        if top_picks_data:
            lines = ["🌟 **Today's Top 3 AI Intraday Stock Recommendations**:"]
            for i, p in enumerate(top_picks_data, 1):
                lines.append(f"{i}. **{p['display_name']}**: {p['action_title']} @ ₹{p['current_price']:,.2f} | Target: ₹{p['target_1_price']:,.2f} (+{p['target_1_gain_pct']:.1f}%) | SL: ₹{p['stop_loss_price']:,.2f}")
            lines.append("\n💡 *To place any trade safely, click on the action card below or say 'Buy [Stock Name]'.*")
            return "\n".join(lines)

        if market_telemetry:
            return f"🌅 **Market Opening Briefing**:\n• **Mood**: {market_telemetry.get('title')}\n• **NIFTY 50**: ₹{market_telemetry.get('nifty_price', 0):,.2f} ({market_telemetry.get('gap_pct', 0):+.2f}% Gap)\n• **Assessment**: {market_telemetry.get('explanation')}"

        if stock_analysis and stock_analysis.get("status") == "SUCCESS":
            disp = stock_analysis.get("display_name", symbol)
            score = stock_analysis.get("score", 5.0)
            verdict = stock_analysis.get("verdict", "WAIT")
            curr_p = stock_analysis.get("current_price", 0.0)
            t1 = stock_analysis.get("target_1", {}).get("price", curr_p * 1.03)
            sl = stock_analysis.get("stop_loss", {}).get("price", curr_p * 0.98)
            desc = stock_analysis.get("verdict_desc", "")
            return f"📊 **Analysis for {disp}** (`₹{curr_p:,.2f}`):\n• **AI Verdict**: **{verdict}** (Score: `{score}/10`)\n• **Ideal Entry**: Around ₹{curr_p:,.2f}\n• **Target 1**: ₹{t1:,.2f}\n• **Safety Stop-Loss**: ₹{sl:,.2f}\n\n💡 *{desc}*"

        return "👋 Hello! I am your **ApexTrade AI Trading Assistant**. You can ask me:\n• *'How is Tata Motors looking for today?'*\n• *'What is NIFTY doing right now?'*\n• *'Show today's top 3 stock picks'*\n• *'What is my profit today?'*\n• *'Buy ₹25,000 of Reliance with safety stop-loss'*"
