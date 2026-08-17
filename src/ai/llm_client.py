"""
Unified Multi-Model LLM Client for Autonomous Trading Decisions.
Supports Anthropic Claude, Google Gemini, Kimi (Moonshot AI), OpenAI GPT-4o, and DeepSeek.
Features comprehensive, professional error parsing and fault-tolerant fallbacks.
"""

import os
import json
import time
import requests
from typing import Optional

class LLMClient:
    """
    Unified LLM interface for autonomous quantitative decision-making.
    """
    
    SUPPORTED_PROVIDERS = {
        "anthropic": ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "gemini": ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-flash-latest", "gemini-3.1-pro-preview", "gemma-4-26b-a4b-it"],
        "kimi": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"]
    }
    
    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-3.1-flash-lite",
        api_key: Optional[str] = None,
        timeout: int = 15
    ):
        self.provider = provider.lower()
        self.model = model.strip() if model else "gemini-3.1-flash-lite"
        self.timeout = timeout
        self.api_key = api_key.strip() if api_key else self._get_env_api_key(self.provider)
        self.last_call_time = 0.0
        self.min_call_interval = 1.0 # Minimum seconds between calls

    def _get_env_api_key(self, provider: str) -> str:
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "kimi": "MOONSHOT_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY"
        }
        key_name = env_map.get(provider, "")
        val = os.getenv(key_name, "")
        if not val:
            try:
                from src.utils.storage import load_ai_settings
                settings = load_ai_settings()
                if settings.get("provider") == provider:
                    val = settings.get("api_key", "")
            except Exception:
                pass
        return val

    def is_configured(self) -> bool:
        """Check if API key is present."""
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def test_connection(self) -> tuple[bool, str]:
        """Test API key connectivity with a lightweight ping and clean error formatting."""
        if not self.is_configured():
            return False, f"API Key is empty or missing. Please enter your {self.provider.upper()} API key."
            
        try:
            res = self.generate_completion(
                system_prompt="You are an Indian stock market assistant. Respond with valid JSON only.",
                user_prompt='Return strictly: {"status": "ok", "provider": "' + self.provider + '"}'
            )
            if res and "status" in res:
                return True, f"Connection Verified: Connected to {self.provider.upper()} ({self.model}) successfully!"
            return False, f"Unexpected response format from {self.provider.upper()}: {res[:150]}"
        except Exception as e:
            return False, self._format_error_message(e)

    def _format_error_message(self, exception: Exception) -> str:
        """Translates raw exceptions into clear, professional, actionable explanations."""
        err_str = str(exception)
        
        if "401" in err_str or "unauthorized" in err_str.lower() or "invalid api key" in err_str.lower() or "authentication" in err_str.lower():
            return f"Authentication Failed (401): The provided {self.provider.upper()} API key is invalid or expired. Please check your key in the provider console."
        elif "404" in err_str or "not found" in err_str.lower():
            return f"Model Not Found (404): The model '{self.model}' is not accessible on your {self.provider.upper()} plan. Please check the model name."
        elif "429" in err_str or "quota" in err_str.lower() or "rate limit" in err_str.lower():
            return f"Usage Limit Exceeded (429): You have hit your {self.provider.upper()} quota or rate limits. Please check your billing or quota credits."
        elif "timeout" in err_str.lower() or "timed out" in err_str.lower():
            return f"Connection Timeout: {self.provider.upper()} servers did not respond within {self.timeout} seconds. Please check your internet connection."
        elif "500" in err_str or "503" in err_str or "service unavailable" in err_str.lower():
            return f"Provider Service Outage (503): {self.provider.upper()} API is experiencing temporary downtime. Please retry in a few moments."
        else:
            return f"API Connection Error: {err_str}"

    def generate_completion(self, system_prompt: str, user_prompt: str) -> str:
        """
        Execute completion with selected provider and return raw text response.
        Enforces rate limiting, timeouts, and clean error handling.
        """
        if not self.is_configured():
            raise ValueError(f"API Key is missing for {self.provider.upper()}. Please provide a valid API key.")
            
        # Rate limit throttling
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
            
        self.last_call_time = time.time()
        
        try:
            if self.provider == "anthropic":
                return self._call_anthropic(system_prompt, user_prompt)
            elif self.provider == "gemini":
                return self._call_gemini(system_prompt, user_prompt)
            elif self.provider == "kimi" or self.provider == "moonshot":
                return self._call_openai_compatible(
                    base_url="https://api.moonshot.cn/v1",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            elif self.provider == "openai":
                return self._call_openai_compatible(
                    base_url="https://api.openai.com/v1",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            elif self.provider == "deepseek":
                return self._call_openai_compatible(
                    base_url="https://api.deepseek.com/v1",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            # Re-raise with clean explanation
            formatted_msg = self._format_error_message(e)
            raise RuntimeError(formatted_msg) from None

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Direct Anthropic Claude Messages API call."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model if self.model else "claude-3-7-sonnet-20250219",
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        content = data.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "")
        return ""

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Calls Google Gemini API with multi-model fallback across active Gemini 3.x and Flash tiers."""
        primary_model = self.model if self.model else "gemini-3.1-flash-lite"
        fallback_models = [
            primary_model,
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-3.1-pro-preview",
            "gemma-4-26b-a4b-it"
        ]
        
        # Deduplicate while preserving order
        candidate_models = []
        for m in fallback_models:
            if m and m not in candidate_models:
                candidate_models.append(m)

        last_error = None
        for model_name in candidate_models:
            # 1. Try google.genai SDK
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=self.api_key)
                prompt = f"{system_prompt}\n\nTask:\n{user_prompt}"
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                if response.text and len(response.text.strip()) > 2:
                    return response.text
            except Exception as e:
                last_error = e
                
            # 2. Direct HTTP Fallback
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [
                        {"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
                    ],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.1
                    }
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text and len(text.strip()) > 2:
                            return text
                else:
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                last_error = e

            # Brief backoff before next candidate
            time.sleep(0.4)

        if last_error:
            raise last_error
        return ""

    def _call_openai_compatible(self, base_url: str, system_prompt: str, user_prompt: str) -> str:
        """Calls OpenAI, Kimi (Moonshot), or DeepSeek via OpenAI-compatible endpoints."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        choices = data.get("choices", [])
        if choices and len(choices) > 0:
            return choices[0].get("message", {}).get("content", "")
        return ""
