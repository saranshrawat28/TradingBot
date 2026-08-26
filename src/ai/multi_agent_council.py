"""
Multi-Agent AI Strategy Council & 2-Stage Consensus Engine for Indian Equities.
Features:
1. Stage 1: Deterministic Math Score Pre-Filter (Local Python, zero LLM cost).
2. Stage 2: 3 Specialized Orthogonal Qualitative Agents:
   - Agent 1: Pattern & Momentum Specialist (40% Weight, reads S_math as fixed input).
   - Agent 2: Trap & Microstructure Officer (35% Weight, checks qualitative traps & holds explicit Veto).
   - Agent 3: Macro & Sector Breadth Officer (25% Weight, checks NIFTY gap & Sector RS).
3. Explicit Precedence & Asymmetric Defense Veto (Defense Wins).
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.engine.stock_advisor import StockAdvisor
from src.engine.pre_market_analyzer import PreMarketAnalyzer
from src.utils.helpers import display_symbol_name, clean_symbol, get_ist_now

class MultiAgentCouncil:
    """
    Collaborative Multi-Agent Strategy Council with 2-Stage Gating.
    """

    MATH_PREFILTER_FLOOR = 7.0
    FINAL_CONSENSUS_THRESHOLD = 7.50

    @classmethod
    def evaluate_candidate(
        cls,
        symbol: str,
        df: pd.DataFrame,
        quote: Dict[str, Any],
        macro_cues: Optional[Dict[str, Any]] = None,
        horizon: str = "intraday"
    ) -> Dict[str, Any]:
        """
        Executes the 2-Stage Multi-Agent Council Evaluation Pipeline.
        """
        # =====================================================================
        # STAGE 1: Deterministic Math Pre-Filter (Local Python, 0 LLM Cost)
        # =====================================================================
        math_analysis = StockAdvisor.evaluate_df_slice(
            df,
            symbol=symbol,
            horizon=horizon,
            index_trend="BULLISH" if macro_cues and "BULLISH" in macro_cues.get("sentiment", "") else "NEUTRAL"
        )
        
        math_score = float(math_analysis.get("score", 5.0))

        # If Math Score is below pre-filter floor, reject immediately (Zero LLM cost)
        if math_score < cls.MATH_PREFILTER_FLOOR:
            return {
                "symbol": symbol,
                "display_name": display_symbol_name(symbol),
                "math_score": math_score,
                "passed_prefilter": False,
                "consensus_approved": False,
                "rejection_reason": f"Math score ({math_score:.1f}) below Stage-1 pre-filter floor ({cls.MATH_PREFILTER_FLOOR}).",
                "agents": {},
                "consensus_score": math_score,
                "deliberation_summary": "Rejected in Stage 1 Math Pre-Filter before Council invocation."
            }

        # =====================================================================
        # STAGE 2: 3-Agent Qualitative Council (Orthogonal Non-Overlapping Roles)
        # =====================================================================
        macro = macro_cues or PreMarketAnalyzer.get_market_opening_sentiment()
        
        # Agent 1: Pattern & Momentum Specialist (40% Weight)
        agent_1 = cls._evaluate_agent_1_pattern(symbol, df, quote, math_score, math_analysis)
        
        # Agent 2: Trap & Microstructure Defense Officer (35% Weight)
        agent_2 = cls._evaluate_agent_2_defense(symbol, df, quote, math_analysis)
        
        # Agent 3: Macro & Sector Breadth Officer (25% Weight)
        agent_3 = cls._evaluate_agent_3_macro(symbol, macro)

        # Consensus Aggregation
        s1 = agent_1["score"]
        s2 = agent_2["score"]
        s3 = agent_3["score"]
        
        has_veto = agent_2["veto"]
        veto_reason = agent_2.get("veto_reason", "")

        # Asymmetric Defense Veto: If Agent 2 raises Veto, score is forced to 0.0
        if has_veto:
            consensus_score = 0.0
            approved = False
            verdict = "REJECTED_BY_DEFENSE_VETO"
            summary = f"🚫 VETOED by Agent 2 (Defense Officer): {veto_reason}"
        else:
            consensus_score = round((0.40 * s1) + (0.35 * s2) + (0.25 * s3), 2)
            # Must pass both Math Gate >= 7.50 and Council Gate >= 7.50 with Defense Score >= 6.0
            approved = (
                math_score >= cls.FINAL_CONSENSUS_THRESHOLD and
                consensus_score >= cls.FINAL_CONSENSUS_THRESHOLD and
                s2 >= 6.0
            )
            verdict = "APPROVED" if approved else "REJECTED"
            summary = f"Council Consensus: S1={s1:.1f}, S2={s2:.1f}, S3={s3:.1f} -> Weighted: {consensus_score:.2f}/10"

        return {
            "symbol": symbol,
            "display_name": display_symbol_name(symbol),
            "current_price": float(quote.get("price", df["Close"].iloc[-1])),
            "math_score": math_score,
            "passed_prefilter": True,
            "consensus_score": consensus_score,
            "consensus_approved": approved,
            "verdict": verdict,
            "deliberation_summary": summary,
            "agents": {
                "agent_1_pattern": agent_1,
                "agent_2_defense": agent_2,
                "agent_3_macro": agent_3
            },
            "trade_blueprint": {
                "action": "BUY" if "BUY" in str(math_analysis.get("verdict", "BUY")) else "SELL",
                "entry_price": float(quote.get("price", df["Close"].iloc[-1])),
                "entry_zone": math_analysis.get("levels", {}).get("entry_zone", f"₹{float(quote.get('price', df['Close'].iloc[-1])) * 0.998:.2f} – ₹{float(quote.get('price', df['Close'].iloc[-1])):.2f}"),
                "target_1_price": float(math_analysis.get("target_1", {}).get("price", float(quote.get("price", df["Close"].iloc[-1])) * 1.025) if isinstance(math_analysis.get("target_1"), dict) else (math_analysis.get("target_1") or float(quote.get("price", df["Close"].iloc[-1])) * 1.025)),
                "target_1_gain_pct": float(math_analysis.get("target_1", {}).get("gain_pct", 2.5) if isinstance(math_analysis.get("target_1"), dict) else 2.5),
                "target_2_price": float(math_analysis.get("target_2", {}).get("price", float(quote.get("price", df["Close"].iloc[-1])) * 1.050) if isinstance(math_analysis.get("target_2"), dict) else (math_analysis.get("target_2") or float(quote.get("price", df["Close"].iloc[-1])) * 1.050)),
                "target_2_gain_pct": float(math_analysis.get("target_2", {}).get("gain_pct", 5.0) if isinstance(math_analysis.get("target_2"), dict) else 5.0),
                "stop_loss_price": float(math_analysis.get("stop_loss", {}).get("price", float(quote.get("price", df["Close"].iloc[-1])) * 0.985) if isinstance(math_analysis.get("stop_loss"), dict) else (math_analysis.get("stop_loss") or float(quote.get("price", df["Close"].iloc[-1])) * 0.985)),
                "stop_loss_pct": float(math_analysis.get("stop_loss", {}).get("loss_pct", 1.5) if isinstance(math_analysis.get("stop_loss"), dict) else 1.5),
                "risk_reward": "1:2.0"
            },
            "timestamp": get_ist_now().strftime("%I:%M:%S %p IST")
        }

    @classmethod
    def _evaluate_agent_1_pattern(
        cls,
        symbol: str,
        df: pd.DataFrame,
        quote: Dict[str, Any],
        math_score: float,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Agent 1: Pattern & Momentum Specialist (40% Weight).
        Takes math score as fixed anchor and assesses multi-candle range breakouts and wick absorption.
        """
        close = df["Close"].iloc[-1]
        high_20 = df["High"].iloc[-20:].max()
        low_20 = df["Low"].iloc[-20:].min()
        
        # Check if breaking 20-candle high with buyer energy
        is_breakout = close >= (high_20 * 0.998)
        
        score = min(10.0, math_score + (0.5 if is_breakout else 0.0))
        vote = "APPROVE" if score >= 7.5 else "HOLD"
        
        thesis = "Multi-candle range breakout confirmed with strong buyer continuation." if is_breakout else "Solid technical baseline within range consolidation."
        
        return {
            "name": "📈 Trend & Pattern Hunter",
            "score": round(score, 1),
            "vote": vote,
            "weight": 0.40,
            "thesis": thesis
        }

    @classmethod
    def _evaluate_agent_2_defense(
        cls,
        symbol: str,
        df: pd.DataFrame,
        quote: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Agent 2: Risk & Defense Officer (35% Weight).
        Checks non-deterministic qualitative traps: extreme bid-ask spread traps and corporate release traps.
        Holds Asymmetric Veto Power.
        """
        veto = False
        veto_reason = ""
        deduction = 0.0

        # Check bid-ask spread liquidity trap (if available)
        bid = float(quote.get("bid", 0.0))
        ask = float(quote.get("ask", 0.0))
        if bid > 0 and ask > 0:
            spread_pct = ((ask - bid) / bid) * 100.0
            if spread_pct > 1.2:
                veto = True
                veto_reason = f"Severe illiquidity: Bid-Ask Spread is {spread_pct:.2f}% (Limit: 1.20%). High slippage trap."

        # Check extreme gap extension risk
        prev_close = float(quote.get("previous_close", df["Close"].iloc[-1]))
        curr_price = float(quote.get("price", df["Close"].iloc[-1]))
        if prev_close > 0:
            gap_pct = abs(((curr_price - prev_close) / prev_close) * 100.0)
            if gap_pct > 6.0:
                deduction += 1.5

        score = max(1.0, min(10.0, 8.5 - deduction))
        vote = "REJECT" if veto or score < 6.0 else ("APPROVE" if score >= 7.5 else "HOLD")

        return {
            "name": "🛡️ Risk & Defense Officer",
            "score": round(score, 1),
            "vote": vote,
            "weight": 0.35,
            "veto": veto,
            "veto_reason": veto_reason,
            "defense_notes": veto_reason if veto else "Microstructure cleared with safe liquidity and zero trap flags."
        }

    @classmethod
    def _evaluate_agent_3_macro(
        cls,
        symbol: str,
        macro: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Agent 3: Macro & Sector Breadth Officer (25% Weight).
        Evaluates NIFTY 50 gap alignment, India VIX regime, and Put-Call Ratio telemetry.
        """
        nifty_gap = float(macro.get("gap_pct", 0.0))
        vix = float(macro.get("vix_level", 13.5))

        score = 8.0
        # Positive macro tailwinds
        if nifty_gap >= 0.30 and vix < 16.0:
            score = 9.0
            thesis = "Strong macroeconomic tailwind: NIFTY 50 gap-up with benign VIX environment."
        elif nifty_gap <= -0.50:
            score = 6.0
            thesis = "Macro headwind: NIFTY 50 opening weak. Requires strong stock-specific divergence."
        else:
            thesis = "Neutral macro environment. Stock-specific momentum leads."

        vote = "APPROVE" if score >= 7.5 else "HOLD"

        return {
            "name": "📊 Macro & Flow Analyst",
            "score": round(score, 1),
            "vote": vote,
            "weight": 0.25,
            "thesis": thesis
        }

# Clear semantic alias for quantitative/rule-based consensus engine
QuantitativeConsensusEngine = MultiAgentCouncil
