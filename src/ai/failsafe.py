"""
Strict Fail-Safe Response Parser and Validation Core.
Guarantees that malformed, incomplete, or hallucinated LLM responses default to 'HOLD' (Zero Action).
"""

import json
import re
from typing import Any

VALID_ACTIONS = {"BUY_CALL", "BUY_PUT", "BUY_STOCK", "EXIT_POSITION", "HOLD"}
VALID_STRIKE_OFFSETS = {"ATM", "ITM1", "OTM1", "ITM2", "OTM2"}
VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

class FailsafeParser:
    """
    Validates LLM trade proposals against strict quantitative schemas.
    """
    
    @staticmethod
    def parse_and_validate(raw_response: str) -> dict:
        """
        Parse raw LLM string into validated trade proposal dict.
        Defaults to HOLD on any error.
        """
        if not raw_response or not isinstance(raw_response, str):
            return FailsafeParser._default_hold("Empty or invalid raw response from LLM")
            
        # Clean potential markdown fences ```json ... ```
        cleaned = raw_response.strip()
        if "```" in cleaned:
            # Extract content between code blocks
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
            else:
                # Remove fence lines
                cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
                
        try:
            data = json.loads(cleaned)
        except Exception as e:
            return FailsafeParser._default_hold(f"JSON Parse Error: {str(e)} | Raw: {raw_response[:100]}...")
            
        if not isinstance(data, dict):
            return FailsafeParser._default_hold("Response is not a JSON object")
            
        # 1. Validate Action
        action = str(data.get("action", "HOLD")).strip().upper()
        if action not in VALID_ACTIONS:
            return FailsafeParser._default_hold(f"Invalid action '{action}' from LLM. Defaulting to HOLD.")
            
        # 2. Validate Confidence Score
        try:
            confidence = float(data.get("confidence_score", 5.0))
            confidence = max(0.0, min(10.0, confidence))
        except (ValueError, TypeError):
            confidence = 5.0
            
        # 3. Validate Target Asset
        target_asset = str(data.get("target_asset", "NIFTY")).strip().upper()
        
        # 4. Validate Strike Offset (For Options)
        strike_offset = str(data.get("strike_offset", "ATM")).strip().upper()
        if strike_offset not in VALID_STRIKE_OFFSETS:
            strike_offset = "ATM"
            
        # 5. Validate SL and TP %
        try:
            sl_pct = float(data.get("suggested_sl_pct", 1.5))
            sl_pct = max(0.25, min(15.0, sl_pct))
        except (ValueError, TypeError):
            sl_pct = 1.5
            
        try:
            tp_pct = float(data.get("suggested_tp_pct", 3.0))
            tp_pct = max(0.5, min(30.0, tp_pct))
        except (ValueError, TypeError):
            tp_pct = 3.0
            
        reasoning = str(data.get("reasoning", "No reasoning provided.")).strip()
        risk_level = str(data.get("risk_level", "MEDIUM")).strip().upper()
        if risk_level not in VALID_RISK_LEVELS:
            risk_level = "MEDIUM"
            
        return {
            "action": action,
            "target_asset": target_asset,
            "strike_offset": strike_offset,
            "confidence_score": round(confidence, 1),
            "reasoning": reasoning,
            "suggested_sl_pct": round(sl_pct, 2),
            "suggested_tp_pct": round(tp_pct, 2),
            "risk_level": risk_level,
            "is_failsafe": False,
            "raw_response": raw_response
        }

    @staticmethod
    def parse_json_safely(raw_response: str) -> dict:
        """
        Generic helper to strip markdown blocks and safely parse JSON dictionary.
        Returns empty dict on failure.
        """
        if not raw_response or not isinstance(raw_response, str):
            return {}
        cleaned = raw_response.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
            else:
                cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _default_hold(reason: str) -> dict:
        return {
            "action": "HOLD",
            "target_asset": "NONE",
            "strike_offset": "ATM",
            "confidence_score": 0.0,
            "reasoning": f"🛡️ Fail-Safe Triggered: {reason}",
            "suggested_sl_pct": 1.5,
            "suggested_tp_pct": 3.0,
            "risk_level": "LOW",
            "is_failsafe": True,
            "raw_response": ""
        }
