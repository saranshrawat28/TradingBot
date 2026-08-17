"""
Institutional Market Context Builder and Prompt Generator for LLM Decision Engine.
Constructs rich, token-efficient market snapshots for Nifty, BankNifty, and Indian Equities.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from src.utils.helpers import get_ist_now
from src.strategies.indicators import (
    calculate_ema, calculate_rsi, calculate_macd,
    calculate_supertrend, calculate_atr, calculate_bollinger_bands
)

class MarketPrompter:
    """
    Builds structured market context and institutional decision prompts for LLMs.
    """
    
    SYSTEM_PROMPT = """You are an institutional Indian Market Trading AI (NSE/NFO).
Evaluate market structure, indicators, and risk. Default to HOLD in chop. Require min 1:2 R:R, conf >= 7.5.
Output strictly JSON:
{
  "action": "BUY_CALL" | "BUY_PUT" | "BUY_STOCK" | "EXIT_POSITION" | "HOLD",
  "target_asset": "SYMBOL",
  "strike_offset": "ATM" | "ITM1" | "OTM1",
  "confidence_score": <float 1.0-10.0>,
  "reasoning": "<1-sentence rationale>",
  "suggested_sl_pct": <float>,
  "suggested_tp_pct": <float>,
  "risk_level": "LOW" | "MEDIUM" | "HIGH"
}"""

    @staticmethod
    def build_market_prompt(
        symbol: str,
        live_quote: dict,
        df_5m: pd.DataFrame,
        df_15m: Optional[pd.DataFrame] = None,
        active_positions: list = None,
        account_summary: dict = None
    ) -> str:
        """
        Constructs token-optimized dense market context for the LLM.
        """
        now_ist = get_ist_now().strftime("%H:%M:%S IST")
        active_positions = active_positions or []
        account_summary = account_summary or {"capital": 100000, "daily_pnl": 0.0, "open_legs": 0}
        
        ltp = live_quote.get("price", 0.0)
        chg_pct = live_quote.get("change_pct", 0.0)
        day_high = live_quote.get("high", ltp)
        day_low = live_quote.get("low", ltp)
        prev_close = live_quote.get("previous_close", ltp)
        volume = live_quote.get("volume", 0)
        
        # Calculate technical indicators on 5m timeframe
        tech_5m = {}
        if not df_5m.empty and len(df_5m) >= 20:
            closes = df_5m["Close"]
            highs = df_5m["High"]
            lows = df_5m["Low"]
            volumes = df_5m["Volume"] if "Volume" in df_5m.columns else pd.Series(1, index=df_5m.index)
            
            ema9 = calculate_ema(closes, 9).iloc[-1]
            ema21 = calculate_ema(closes, 21).iloc[-1]
            ema50 = calculate_ema(closes, 50).iloc[-1] if len(df_5m) >= 50 else ema21
            rsi = calculate_rsi(closes, 14).iloc[-1]
            macd_df = calculate_macd(closes)
            macd_hist = macd_df["MACD_Hist"].iloc[-1]
            atr = calculate_atr(highs, lows, closes, 14).iloc[-1]
            st_df, st_dir = calculate_supertrend(highs, lows, closes, 10, 3.0)
            st_trend = "Bullish" if st_dir.iloc[-1] == 1 else "Bearish"
            adx_s, p_di, m_di = calculate_adx(highs, lows, closes, 14)
            last_adx = float(adx_s.iloc[-1])
            regime = "Trending" if last_adx >= 25 else ("Chop" if last_adx < 20 else "Transitional")
            vwap_val = float(calculate_vwap(highs, lows, closes, volumes).iloc[-1])
            
            tech_5m = {
                "ema9": round(ema9, 2),
                "ema21": round(ema21, 2),
                "ema50": round(ema50, 2),
                "rsi": round(rsi, 1),
                "macd_hist": round(macd_hist, 2),
                "atr": round(atr, 2),
                "supertrend": st_trend,
                "adx": round(last_adx, 1),
                "regime": regime,
                "vwap": round(vwap_val, 2)
            }
            
        # Determine ATM strike for Index
        is_index = any(idx in symbol.upper() for idx in ["NIFTY", "BANKNIFTY", "FINNIFTY", "^NSEI", "^NSEBANK"])
        atm_strike = int(round(ltp / 50.0) * 50) if "BANKNIFTY" not in symbol.upper() else int(round(ltp / 100.0) * 100)
        
        prompt = f"""TIME: {now_ist} | INSTRUMENT: {symbol} {"(Index F&O)" if is_index else "(Equity)"} | LTP: ₹{ltp:,.2f} ({chg_pct:+.2f}%) | Day: ₹{day_low:,.2f}-₹{day_high:,.2f} | ATM: {atm_strike}
5M TECH: Regime: {tech_5m.get('regime', 'N/A')} (ADX {tech_5m.get('adx', 'N/A')}) | VWAP: ₹{tech_5m.get('vwap', 'N/A')} | EMAs: 9:₹{tech_5m.get('ema9','N/A')}, 21:₹{tech_5m.get('ema21','N/A')}, 50:₹{tech_5m.get('ema50','N/A')} | RSI: {tech_5m.get('rsi','N/A')} | SuperTrend: {tech_5m.get('supertrend','N/A')} | MACD: {tech_5m.get('macd_hist','N/A')} | ATR: ₹{tech_5m.get('atr','N/A')}
RISK: Cap: ₹{account_summary.get('capital', 100000):,.0f} | Day PnL: ₹{account_summary.get('daily_pnl', 0.0):+,.0f} | Open Legs: {len(active_positions)}
TASK: Evaluate trade setup in valid JSON."""

        return prompt
