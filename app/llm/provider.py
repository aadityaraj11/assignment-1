"""
app/llm/provider.py
===================
Abstracts LLM inference for metadata extraction.
Supports Gemini REST API (HTTPS over requests) and Mock providers.
"""
from abc import ABC, abstractmethod
import json
import logging
import os
import time
from typing import Optional, Type, Tuple, Dict, Any
import requests
from dotenv import load_dotenv

from pydantic import BaseModel

load_dotenv()

log = logging.getLogger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        pass


class MockProvider(LLMProvider):
    """
    Mock provider for offline testing and pipeline validation.
    Always returns a valid schema filled with dummy generic answers.
    """
    def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        log.info("MockProvider: Returning mock structured JSON.")
        return {
            "agreement_value": "12000",
            "agreement_start_date": "01.01.2023",
            "agreement_end_date": "31.12.2023",
            "renewal_notice_days": "30",
            "party_one": "John Doe",
            "party_two": "Jane Smith"
        }


class GeminiRESTProvider(LLMProvider):
    """
    Direct REST-based Gemini Provider using standard HTTPS requests.
    Bypasses SDK transport issues with explicit finite timeouts.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        # connect timeout: 10s, read timeout: 60s
        self.timeout = (10.0, 60.0)
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.session = requests.Session()

    def _sanitize(self, text: str) -> str:
        if self.api_key and isinstance(text, str):
            return text.replace(self.api_key, "[REDACTED]")
        return str(text)

    def check_availability(self) -> Tuple[bool, int, str]:
        """Performs a lightweight availability check for the configured model."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
        headers = {"x-goog-api-key": self.api_key}
        try:
            resp = self.session.get(url, headers=headers, timeout=(10.0, 10.0))
            if resp.status_code == 200:
                return True, resp.status_code, "OK"
            else:
                sanitized_msg = self._sanitize(resp.text)
                return False, resp.status_code, sanitized_msg
        except Exception as e:
            return False, 0, self._sanitize(str(e))

    def _clean_json_text(self, text: str) -> str:
        """Strips markdown code blocks around JSON response if present."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned

    def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        log.info("GeminiRESTProvider: Calling %s via HTTPS REST", self.model)
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "agreement_value": {"type": "STRING", "nullable": True},
                        "agreement_start_date": {"type": "STRING", "nullable": True},
                        "agreement_end_date": {"type": "STRING", "nullable": True},
                        "renewal_notice_days": {"type": "STRING", "nullable": True},
                        "party_one": {"type": "STRING", "nullable": True},
                        "party_two": {"type": "STRING", "nullable": True}
                    },
                    "required": [
                        "agreement_value",
                        "agreement_start_date",
                        "agreement_end_date",
                        "renewal_notice_days",
                        "party_one",
                        "party_two"
                    ]
                }
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        attempts = 0
        max_attempts = 2
        last_exception = None

        while attempts < max_attempts:
            attempts += 1
            start_time = time.time()
            try:
                resp = self.session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                latency = round(time.time() - start_time, 3)
                resp_size = len(resp.content)
                
                if resp.status_code == 200:
                    log.info(
                        "Gemini REST Success — Model: %s, Status: %d, Latency: %.3fs, Size: %d bytes",
                        self.model, resp.status_code, latency, resp_size
                    )
                    resp_json = resp.json()
                    candidates = resp_json.get("candidates", [])
                    if not candidates:
                        raise ValueError("Gemini returned empty candidates list.")
                    
                    raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    cleaned_text = self._clean_json_text(raw_text)
                    
                    try:
                        parsed_dict = json.loads(cleaned_text)
                    except json.JSONDecodeError as e:
                        log.error("Gemini REST JSON Decode Error: %s", e)
                        raise ValueError(f"Failed to decode JSON from Gemini response: {e}")
                    
                    if not isinstance(parsed_dict, dict):
                        raise ValueError(f"Gemini response did not parse into a dictionary: {type(parsed_dict)}")
                        
                    return parsed_dict
                else:
                    sanitized_body = self._sanitize(resp.text)
                    log.error(
                        "Gemini REST HTTP Error — Model: %s, Status: %d, Latency: %.3fs, Error: %s",
                        self.model, resp.status_code, latency, sanitized_body
                    )
                    err_msg = f"Gemini API HTTP Error {resp.status_code}: {sanitized_body}"
                    if resp.status_code == 429:
                        log.warning("Gemini REST Rate Limit (429) hit — waiting 15s before retry...")
                        last_exception = RuntimeError(err_msg)
                        if attempts < max_attempts:
                            time.sleep(15.0)
                            continue
                    elif 400 <= resp.status_code < 500:
                        raise RuntimeError(err_msg)
                    
                    last_exception = RuntimeError(err_msg)
                    if attempts < max_attempts:
                        log.warning("Retrying request after status %d...", resp.status_code)
                        time.sleep(1.0)
            except requests.Timeout as e:
                latency = round(time.time() - start_time, 3)
                sanitized_err = self._sanitize(str(e))
                log.error("Gemini REST Timeout — Model: %s, Latency: %.3fs, Error: %s", self.model, latency, sanitized_err)
                last_exception = RuntimeError(f"Gemini API Request Timeout: {sanitized_err}")
                if attempts < max_attempts:
                    log.warning("Retrying request after timeout...")
                    time.sleep(1.0)
            except requests.RequestException as e:
                latency = round(time.time() - start_time, 3)
                sanitized_err = self._sanitize(str(e))
                log.error("Gemini REST Connection Error — Model: %s, Latency: %.3fs, Error: %s", self.model, latency, sanitized_err)
                last_exception = RuntimeError(f"Gemini API Connection Error: {sanitized_err}")
                if attempts < max_attempts:
                    log.warning("Retrying request after connection error...")
                    time.sleep(1.0)
            except (ValueError, RuntimeError) as e:
                raise e

        if last_exception:
            raise last_exception
        raise RuntimeError("Gemini REST provider failed without explicit exception.")


# Maintain backward compatibility alias
GeminiProvider = GeminiRESTProvider


def get_provider() -> LLMProvider:
    provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider_name == "mock":
        return MockProvider()
    elif provider_name == "gemini":
        return GeminiRESTProvider()
    else:
        raise ValueError(f"Unknown LLM Provider: '{provider_name}'. Must be one of ['gemini', 'mock'].")

