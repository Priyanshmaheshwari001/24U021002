"""Gemini LLM Client for Humour Classification.

Provides a reusable classification function using Google's official `google-genai` SDK:
- Deterministic output (temperature = 0, application/json response MIME type).
- Safe JSON parsing with fallback to label = -1 on formatting errors.
- Robust retry with exponential backoff (tenacity) for rate limits and transient errors.
- SQLite-based disk cache in results/ to avoid redundant API calls and save quota.
- Configurable model name and rate-limit sleep delay.
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv(override=True)

# ==============================================================================
# Model Configuration (Change this variable to switch models)
# ==============================================================================
# Fast and cost-efficient stable model with active free-tier quota (500 RPD, 15 RPM)
MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Delay (in seconds) between live API requests to stay within 15 RPM free rate limit
REQUEST_DELAY_SECONDS: float = 5.0

# Default prompt template if none is supplied
DEFAULT_PROMPT_TEMPLATE: str = """You are an NLP humor detection assistant.
Determine whether the following text is humorous (1) or not humorous (0).

Respond ONLY with a JSON object in this exact format:
{
  "label": 0 or 1,
  "reason": "Brief explanation of the classification"
}

Text to classify:
{text}
"""

# SQLite Cache Location
CACHE_DB_PATH: str = os.path.join("results", "llm_cache.sqlite")


# ==============================================================================
# Local SQLite Cache
# ==============================================================================
class LLMCache:
    """Persistent SQLite cache for storing LLM prompts and responses."""

    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_template TEXT NOT NULL,
                    text TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    raw_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def _generate_key(model: str, prompt_template: str, text: str) -> str:
        payload = f"{model.strip()}:::{prompt_template.strip()}:::{text.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt_template: str, text: str) -> Optional[Dict[str, Any]]:
        key = self._generate_key(model, prompt_template, text)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT label, reason, raw_response FROM response_cache WHERE cache_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "label": int(row[0]),
                    "reason": row[1],
                    "raw_response": row[2],
                    "cached": True,
                }
        return None

    def set(
        self,
        model: str,
        prompt_template: str,
        text: str,
        label: int,
        reason: str,
        raw_response: str,
    ) -> None:
        key = self._generate_key(model, prompt_template, text)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO response_cache 
                (cache_key, model, prompt_template, text, label, reason, raw_response)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (key, model, prompt_template, text, int(label), reason, raw_response),
            )
            conn.commit()


# Shared singleton cache instance
_cache_instance = LLMCache()


# ==============================================================================
# Client & API Utilities
# ==============================================================================
def get_api_key() -> str:
    """Retrieve GEMINI_API_KEY from environment or .env file.
    
    Accepts any non-empty string, supporting standard keys as well as 'AQ.' prefixed keys.
    """
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    placeholder_values = {"", "your_actual_gemini_api_key", "your_key_here", "your_gemini_api_key_here"}

    if not api_key or api_key.strip() in placeholder_values:
        raise ValueError(
            "GEMINI_API_KEY is missing or empty in .env!\n"
            "Please open the '.env' file in the project root and add your API key:\n"
            "    GEMINI_API_KEY=\"your_key_here\""
        )
    return api_key.strip()


_client_instance: Optional[genai.Client] = None


def get_gemini_client() -> genai.Client:
    """Initialize or reuse a singleton Google GenAI Client."""
    global _client_instance
    if _client_instance is None:
        api_key = get_api_key()
        _client_instance = genai.Client(api_key=api_key)
    return _client_instance


def _is_retryable_exception(exc: BaseException) -> bool:
    """Determine whether an error is transient and should be retried."""
    if isinstance(exc, errors.APIError):
        # Do not retry client configuration or authentication errors
        msg = str(exc).lower()
        if exc.code in (400, 401, 403) or "api key not valid" in msg or "invalid argument" in msg:
            return False
        # Retry on rate limit (429) or temporary server errors (500, 502, 503, 504)
        if exc.code in (429, 500, 502, 503, 504) or any(
            token in msg for token in ("resource_exhausted", "quota", "rate limit")
        ):
            print(f"[RETRY] Transient API error encountered ({exc.code}): {exc.message}. Retrying...")
            return True
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        print(f"[RETRY] Network/connection error encountered ({type(exc).__name__}). Retrying...")
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable_exception),
    wait=wait_exponential(multiplier=2.0, min=5, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_api_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
):
    """Invoke client.models.generate_content with exponential backoff."""
    return client.models.generate_content(model=model, contents=contents, config=config)


def parse_llm_json(raw_input: Any) -> Dict[str, Any]:
    """Safely parse model output into label and reason dictionary.

    Handles raw string responses or response objects, strips markdown code blocks,
    and returns label = -1 if parsing fails or label cannot be resolved to 0 or 1.
    """
    if raw_input is None:
        return {"label": -1, "reason": "Empty response from LLM"}

    # Extract text if response object or non-string was passed
    if isinstance(raw_input, str):
        raw_text = raw_input
    elif hasattr(raw_input, "text") and raw_input.text:
        try:
            raw_text = raw_input.text
        except (ValueError, AttributeError):
            raw_text = str(raw_input)
    else:
        raw_text = str(raw_input)

    raw_text = raw_text.strip()
    if not raw_text:
        return {"label": -1, "reason": "Empty string response from LLM"}

    # Strip code block fences if present (e.g. ```json ... ``` or ``` ...)
    cleaned = raw_text
    if cleaned.lower().startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except Exception as parse_err:
        # Fallback: attempt to find JSON object substring between { and }
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                return {"label": -1, "reason": f"Failed to parse JSON response: {parse_err} | Raw text: {raw_text[:100]}"}
        else:
            return {"label": -1, "reason": f"Failed to parse JSON response: {parse_err} | Raw text: {raw_text[:100]}"}

    if not isinstance(data, dict):
        return {"label": -1, "reason": f"Expected JSON object, got {type(data).__name__}"}

    raw_label = data.get("label")
    # Accept int 0/1 or string representations
    if raw_label in (0, 1):
        label = int(raw_label)
    elif str(raw_label).strip().lower() in ("1", "humor", "humorous", "funny", "joke"):
        label = 1
    elif str(raw_label).strip().lower() in ("0", "non-humor", "not humorous", "not funny", "plain", "serious"):
        label = 0
    else:
        label = -1

    reason = str(data.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided by model."

    return {"label": label, "reason": reason}


# ==============================================================================
# Main Reusable Classification Function
# ==============================================================================
def classify_humour(
    text: str,
    prompt_template: Any = DEFAULT_PROMPT_TEMPLATE,
    model: str = MODEL_NAME,
    use_cache: bool = True,
    sleep_delay: float = REQUEST_DELAY_SECONDS,
    style: Optional[str] = None,
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify text for humor using Google Gemini API.

    Args:
        text: The text to classify.
        prompt_template: Prompt template string, callable, or strategy name ('zero_shot', 'few_shot', 'cot_style').
        model: Gemini model identifier (defaults to MODEL_NAME).
        use_cache: Whether to query and populate local SQLite cache.
        sleep_delay: Politeness delay (in seconds) between live API requests.
        style: Alias for prompt_template / strategy name (e.g., 'zero_shot', 'few_shot', 'cot_style').
        strategy: Alias for prompt_template / strategy name.

    Returns:
        Dict with keys:
            'label': 1 (humorous), 0 (not humorous), or -1 (parsing/classification failure)
            'reason': short explanatory reason
            'cached': bool indicating whether the response was loaded from cache
            'raw_response': raw output string from the model
    """
    # Resolve strategy or style aliases if passed
    chosen = style or strategy or prompt_template

    # If a strategy name was provided as a string, resolve from prompts.py
    if isinstance(chosen, str) and chosen in ("zero_shot", "few_shot", "cot_style"):
        try:
            from src.prompts import PROMPT_TEMPLATES
            chosen = PROMPT_TEMPLATES[chosen]
        except Exception:
            try:
                from prompts import PROMPT_TEMPLATES
                chosen = PROMPT_TEMPLATES[chosen]
            except Exception:
                pass

    # 1. Resolve template identifier and formatted prompt
    if callable(chosen):
        template_id = getattr(chosen, "__name__", "custom_prompt_fn")
        formatted_prompt = chosen(text)
    elif isinstance(chosen, str):
        template_id = chosen
        if "{text}" in chosen:
            formatted_prompt = chosen.replace("{text}", text)
        elif text in chosen:
            formatted_prompt = chosen
        else:
            formatted_prompt = f"{chosen}\n\nText:\n{text}"
    else:
        template_id = str(chosen)
        formatted_prompt = str(chosen)

    # 2. Check local cache
    if use_cache:
        cached_result = _cache_instance.get(model=model, prompt_template=template_id, text=text)
        if cached_result is not None:
            return cached_result

    # 3. Configure API request with temperature = 0 and application/json
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
    )

    client = get_gemini_client()

    # 4. Execute request with retry mechanism
    try:
        response = _call_api_with_retry(
            client=client,
            model=model,
            contents=formatted_prompt,
            config=config,
        )
        if isinstance(response, str):
            raw_text = response
        elif hasattr(response, "text") and response.text:
            try:
                raw_text = response.text
            except (ValueError, AttributeError):
                raw_text = ""
        else:
            raw_text = str(response) if response is not None else ""
    except errors.APIError as exc:
        msg = str(exc).lower()
        if "api key not valid" in msg or exc.code == 400:
            raise ValueError(
                f"Google Gemini API error: Invalid API key. Please check your GEMINI_API_KEY in .env.\nDetails: {exc.message}"
            ) from exc
        raise

    # 5. Parse JSON safely
    parsed = parse_llm_json(raw_text)
    parsed["cached"] = False
    parsed["raw_response"] = raw_text

    # 6. Save valid parses to local cache
    if use_cache:
        _cache_instance.set(
            model=model,
            prompt_template=template_id,
            text=text,
            label=parsed["label"],
            reason=parsed["reason"],
            raw_response=raw_text,
        )

    # 7. Sleep between live requests to respect quota limits
    if sleep_delay > 0:
        time.sleep(sleep_delay)

    return parsed


# ==============================================================================
# Test Script
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Gemini Humour Classification Client - Test Suite")
    print(f"Active Model: {MODEL_NAME}")
    print("=" * 70)

    # Validate API Key presence before running test
    try:
        key = get_api_key()
        masked_key = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "***"
        print(f"API Key loaded successfully: {masked_key}")
    except Exception as exc:
        print("\n" + "!" * 70)
        print("[API KEY ERROR] Action Required:")
        print(exc)
        print("!" * 70 + "\n")
        sys.exit(1)

    test_sentences = [
        "I told my doctor that I broke my arm in two places. He told me to stop going to those places.",
        "The climate change symposium will convene tomorrow at 9:00 AM in Hall B.",
        "Why don't scientists trust atoms? Because they make up everything!",
    ]

    print("\nRunning classification on 3 test sentences...\n")
    for idx, sentence in enumerate(test_sentences, 1):
        print(f"Test #{idx}:")
        print(f"  Input: \"{sentence}\"")
        try:
            result = classify_humour(text=sentence)
            print(f"  Prediction Label: {result['label']} ({'Humorous' if result['label'] == 1 else 'Not Humorous' if result['label'] == 0 else 'Error'})")
            print(f"  Reason: {result['reason']}")
            print(f"  Cached: {result['cached']}")
        except Exception as exc:
            print(f"  [Error]: {exc}")
            # If the error is due to an invalid key from the API
            if "invalid api key" in str(exc).lower() or "api key not valid" in str(exc).lower():
                print("\n" + "!" * 70)
                print("[API KEY INVALID] Your GEMINI_API_KEY in .env was rejected by Google Gemini.")
                print("Please verify or regenerate your key at: https://aistudio.google.com/app/apikey")
                print("!" * 70)
                sys.exit(1)
        print("-" * 70)