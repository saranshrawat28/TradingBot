"""
Autonomous AI Trading Agent Core Controller.
Orchestrates Market Prompter, LLM Brain (Claude/Kimi/GPT/Gemini), Guardrails, and Broker Execution.
"""

import time
import logging
from datetime import datetime
from typing import Optional, Any
from src.ai.llm_client import LLMClient
from src.ai.market_prompter import MarketPrompter
from src.ai.failsafe import FailsafeParser
from src.ai.calibration import ConfidenceCalibrator
from src.engine.ai_guardrails import AIGuardrails
from src.engine.reconciliation import StateReconciler
from src.engine.stock_advisor import StockAdvisor
from src.utils.storage import log_calibration_entry
from src.data.data_fetcher import get_live_quote, get_historical_data
from src.utils.helpers import get_ist_now

logger = logging.getLogger("AITradingAgent")

class AITradingAgent:
    """
    Autonomous AI Quantitative Agent for Indian Markets with Asymmetric Consensus.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        guardrails: AIGuardrails,
        broker: Any,
        is_live_mode: bool = False,
        min_consensus_threshold: float = 7.5
    ):
        self.llm = llm_client
        self.guardrails = guardrails
        self.broker = broker
        self.is_live_mode = is_live_mode
        self.min_consensus_threshold = min_consensus_threshold
        self.execution_history = []
        self.is_running = False

    def evaluate_and_execute(self, symbol: str) -> dict:
        """
        Execute a complete end-to-end AI decision and risk-managed execution cycle.
        """
        iteration_start = time.time()
        
        # 1. State Reconciliation with Ground Truth Broker
        portfolio_state = StateReconciler.reconcile_with_broker(self.broker)
        
        # 2. Fetch Live Market Telemetry & 5m / 15m historical candles
        live_quote = get_live_quote(symbol)
        ltp = live_quote.get("price", 0.0)
        
        if ltp <= 0:
            return {
                "status": "ERROR",
                "message": f"Could not retrieve live price for {symbol}.",
                "timestamp": get_ist_now().isoformat()
            }
            
        df_5m = get_historical_data(symbol, period="5d", interval="5m")
        df_15m = get_historical_data(symbol, period="1mo", interval="15m")
        
        # 3. Tier 1: Evaluate Quantitative Math Scorer (Orthogonal + ADX Regime)
        math_eval = StockAdvisor.evaluate_df_slice(df_5m, symbol) if not df_5m.empty else {"score": 5.0, "regime": "UNKNOWN"}
        math_score = float(math_eval.get("score", 5.0))
        regime = math_eval.get("regime", "UNKNOWN")

        # 4. Build Institutional Decision Prompt (Raw metrics only, no score leak)
        user_prompt = MarketPrompter.build_market_prompt(
            symbol=symbol,
            live_quote=live_quote,
            df_5m=df_5m,
            df_15m=df_15m,
            active_positions=portfolio_state.get("open_positions", []),
            account_summary=portfolio_state
        )
        
        # 5. Query AI LLM Core (Tier 2 Cognitive Auditor)
        try:
            raw_ai_response = self.llm.generate_completion(
                system_prompt=MarketPrompter.SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
        except Exception as e:
            raw_ai_response = ""
            logger.error(f"LLM generation failed: {e}")
            
        # 6. Strict Fail-Safe Validation
        proposal = FailsafeParser.parse_and_validate(raw_ai_response)
        llm_conf = float(proposal.get("confidence_score", 0.0))
        llm_action = proposal.get("action", "HOLD")
        
        # 7. Asymmetric Consensus & Veto Engine
        # Rule: Order executes ONLY IF Math >= 7.5 AND LLM >= 7.5 AND Guardrails Approve
        disagreement = False
        disagreement_reason = ""
        
        if llm_action in ["BUY_STOCK", "BUY_CALL", "BUY_PUT"]:
            if math_score < self.min_consensus_threshold:
                # LLM attempted to buy, but Math Scorer is weak -> Defense VETO
                proposal["action"] = "HOLD"
                disagreement = True
                disagreement_reason = f"LLM proposed {llm_action} (Conf: {llm_conf}/10), but Math Scorer is below {self.min_consensus_threshold} ({math_score:.1f}/10). Defense vetoes trade."
            elif llm_conf < self.min_consensus_threshold:
                proposal["action"] = "HOLD"
                disagreement = True
                disagreement_reason = f"Math Score is strong ({math_score:.1f}/10), but LLM Confidence is below safety threshold ({llm_conf}/10). Trade blocked."
        elif math_score >= self.min_consensus_threshold and llm_action in ["HOLD", "EXIT_POSITION"]:
            # Math said Buy, but LLM exercised caution -> Defense VETO
            disagreement = True
            disagreement_reason = f"Math Scorer suggested BUY ({math_score:.1f}/10), but LLM exercised caution ({llm_action}). Defense vetoes trade."
            
        # 8. Log Calibration & Divergence Entry
        log_calibration_entry({
            "timestamp": get_ist_now().isoformat(),
            "symbol": symbol,
            "math_score": math_score,
            "llm_confidence": llm_conf,
            "market_regime": regime,
            "proposed_action": llm_action,
            "final_action": proposal.get("action"),
            "disagreement": disagreement,
            "disagreement_reason": disagreement_reason,
            "entry_price": ltp,
            "prompt_version": "v2.0_orthogonal",
            "model_id": f"{self.llm.provider}_{self.llm.model}"
        })
        
        # 9. Evaluate Against Deterministic Risk Guardrails
        market_depth = {"price": ltp, "bid": live_quote.get("low", ltp), "ask": live_quote.get("high", ltp)}
        is_approved, guardrail_reason, sanitized_order = self.guardrails.evaluate_proposal(
            proposal=proposal,
            portfolio_state=portfolio_state,
            current_market_depth=market_depth
        )
        
        # 8. Order Execution if Approved
        execution_result = {"status": "SKIPPED", "message": guardrail_reason}
        
        if is_approved and sanitized_order.get("action") not in ["HOLD", "EXIT"]:
            action = sanitized_order["action"]
            qty = sanitized_order["quantity"]
            sl_pct = sanitized_order["suggested_sl_pct"]
            tp_pct = sanitized_order["suggested_tp_pct"]
            
            # Determine execution symbol (Option contract or Equity)
            if action in ["BUY_CALL", "BUY_PUT"]:
                option_type = "CE" if action == "BUY_CALL" else "PE"
                atm_strike = int(round(ltp / 50.0) * 50) if "BANKNIFTY" not in symbol else int(round(ltp / 100.0) * 100)
                
                # Resolve option symbol
                if hasattr(self.broker, "resolve_option_symbol"):
                    target_contract = self.broker.resolve_option_symbol(symbol, strike=atm_strike, option_type=option_type)
                else:
                    target_contract = f"{symbol}_{atm_strike}_{option_type}"
                    
                # Approximate option premium
                opt_price = max(10.0, round(ltp * 0.015, 2))
                sl_price = round(opt_price * (1 - (sl_pct / 100.0)), 1)
                tp_price = round(opt_price * (1 + (tp_pct / 100.0)), 1)
                
                execution_result = self.broker.place_order(
                    symbol=target_contract,
                    side="BUY",
                    quantity=qty,
                    order_type="MARKET",
                    product="MIS",
                    price=opt_price,
                    sl=sl_price,
                    tp=tp_price,
                    strategy_name=f"AI_{self.llm.provider.upper()}"
                )
            elif action == "BUY_STOCK":
                sl_price = round(ltp * (1 - (sl_pct / 100.0)), 2)
                tp_price = round(ltp * (1 + (tp_pct / 100.0)), 2)
                execution_result = self.broker.place_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=qty,
                    order_type="MARKET",
                    product="MIS",
                    price=ltp,
                    sl=sl_price,
                    tp=tp_price,
                    strategy_name=f"AI_{self.llm.provider.upper()}"
                )
        elif is_approved and sanitized_order.get("action") == "EXIT":
            # Square-off active position
            execution_result = self.broker.square_off_all(reason="AI Exit Proposal")
            
        elapsed_sec = round(time.time() - iteration_start, 2)
        
        telemetry = {
            "timestamp": get_ist_now().strftime("%d %b %Y | %H:%M:%S IST"),
            "symbol": symbol,
            "ltp": ltp,
            "action": proposal.get("action"),
            "confidence": proposal.get("confidence_score"),
            "reasoning": proposal.get("reasoning"),
            "risk_level": proposal.get("risk_level"),
            "guardrail_status": "APPROVED" if is_approved else "BLOCKED",
            "guardrail_reason": guardrail_reason,
            "execution": execution_result,
            "latency_sec": elapsed_sec,
            "provider": self.llm.provider,
            "model": self.llm.model,
            "is_live": self.is_live_mode
        }
        
        self.execution_history.append(telemetry)
        return telemetry

    def execute_radar_opportunity(self, opportunity: dict) -> dict:
        """
        Execute a selected opportunity directly from the AI Market Radar.
        Ensures full guardrail compliance and audit trailing.
        """
        portfolio_state = StateReconciler.reconcile_with_broker(self.broker)
        sym = opportunity.get("symbol", "NIFTY")
        action = opportunity.get("action", "BUY_CALL")
        conf = float(opportunity.get("confidence_score", 0.0))
        entry_p = float(opportunity.get("entry_price", 0.0))
        sl_p = float(opportunity.get("stop_loss", 0.0))
        tp_p = float(opportunity.get("target_1", 0.0))
        
        # Calculate SL/TP percentages
        sl_pct = max(0.5, round(abs(entry_p - sl_p) / entry_p * 100.0, 2)) if entry_p > 0 and sl_p > 0 else 1.5
        tp_pct = max(1.0, round(abs(tp_p - entry_p) / entry_p * 100.0, 2)) if entry_p > 0 and tp_p > 0 else 3.0
        
        proposal = {
            "action": action,
            "target_asset": sym,
            "confidence_score": conf,
            "reasoning": opportunity.get("catalyst_reasoning", "AI Radar High-Probability Opportunity"),
            "suggested_sl_pct": sl_pct,
            "suggested_tp_pct": tp_pct,
            "risk_level": "MEDIUM"
        }
        
        market_depth = {"price": entry_p, "bid": entry_p * 0.998, "ask": entry_p * 1.002}
        is_approved, guardrail_reason, sanitized_order = self.guardrails.evaluate_proposal(
            proposal=proposal,
            portfolio_state=portfolio_state,
            current_market_depth=market_depth,
            enforce_time_cutoff=self.is_live_mode
        )
        
        if not is_approved:
            return {
                "status": "BLOCKED",
                "message": f"Guardrail blocked order: {guardrail_reason}",
                "symbol": sym,
                "action": action
            }
            
        qty = sanitized_order["quantity"]
        opt_contract = opportunity.get("option_contract", "N/A")
        
        if action in ["BUY_CALL", "BUY_PUT"]:
            exec_symbol = opt_contract if opt_contract != "N/A" else f"{sym}_{int(entry_p)}_OPT"
            exec_price = entry_p if entry_p > 0 else 100.0
            order_res = self.broker.place_order(
                symbol=exec_symbol,
                side="BUY",
                quantity=qty,
                order_type="MARKET",
                product="MIS",
                price=exec_price,
                sl=sl_p if sl_p > 0 else exec_price * 0.85,
                tp=tp_p if tp_p > 0 else exec_price * 1.30,
                strategy_name="AI_Radar_Opportunity"
            )
        else:
            order_res = self.broker.place_order(
                symbol=sym,
                side="BUY",
                quantity=qty,
                order_type="MARKET",
                product="MIS",
                price=entry_p,
                sl=sl_p if sl_p > 0 else entry_p * 0.98,
                tp=tp_p if tp_p > 0 else entry_p * 1.05,
                strategy_name="AI_Radar_Opportunity"
            )
            
        return {
            "status": "EXECUTED" if order_res.get("status") in ["FILLED", "SUCCESS"] else "FAILED",
            "symbol": sym,
            "action": action,
            "execution": order_res,
            "time_horizon": opportunity.get("time_horizon"),
            "setup_name": opportunity.get("setup_name")
        }
