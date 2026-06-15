"""
app/llm.py
----------
Thin async wrapper over LLM providers (Groq and Gemini).
Provider is selected via the LLM_PROVIDER env var in config.py.

Design:
  - call_llm(system, user) → str        — primary interface used by agent.py
  - Automatically falls back to the other provider on failure.
  - parse_json_response(text) → dict     — robust JSON extraction from LLM output.
  - Low temperature (0.1) everywhere for deterministic JSON output.
  - asyncio.wait_for enforces the per-call timeout from config.LLM_TIMEOUT.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import config

logger = logging.getLogger(__name__)


# =============================================================================
# JSON extraction helpers
# =============================================================================

def _strip_fences(text: str) -> str:
    """Remove markdown code fences LLMs sometimes wrap JSON in."""
    text = text.strip()
    # ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    """
    Parse JSON from an LLM reply.

    Attempts in order:
      1. Direct parse after stripping markdown fences.
      2. Extract the first {...} block with regex (handles leading/trailing prose).
      3. Raise ValueError with a diagnostic snippet.

    This covers the common LLM failure modes:
      - Wrapping JSON in ```json ... ```
      - Adding a one-line preamble before the JSON
      - Trailing explanation text after the closing }
    """
    cleaned = _strip_fences(text)

    # Attempt 1: clean parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: regex extraction of outermost {...}
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.error("Failed to parse LLM JSON output (first 300 chars): %r", text[:300])
    raise ValueError(f"LLM returned non-JSON output: {text[:200]!r}")


# =============================================================================
# Provider implementations
# =============================================================================

async def _call_groq(system: str, user: str, timeout: float) -> str:
    """
    Call Groq's llama-3.1-8b-instant via the official groq Python SDK.
    Temperature 0.1 for consistent structured output.
    """
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    from groq import AsyncGroq
    client = AsyncGroq(api_key=config.GROQ_API_KEY)

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.1,
            max_tokens=1024,
        ),
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


async def _call_gemini(system: str, user: str, timeout: float) -> str:
    """
    Call Gemini Flash via google-generativeai SDK.
    System prompt is passed as system_instruction (supported on Flash/Pro).
    run_in_executor wraps the synchronous SDK call so it doesn't block the loop.
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    
    print("DEBUG MODEL =", config.GEMINI_MODEL)
    
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=system,
    )

    resp = await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content(
                user,
                generation_config={"temperature": 0.1, "max_output_tokens": 1024},
            ),
        ),
        timeout=timeout,
    )
    return resp.text or ""


# =============================================================================
# Public interface
# =============================================================================

async def call_llm(
    system: str,
    user: str,
    timeout: float | None = None,
) -> str:
    """
    Send a single-turn request to the configured LLM and return the text reply.

    Falls back to the other provider if the primary fails, so that a Groq
    rate-limit or outage doesn't kill every /chat request.

    Parameters
    ----------
    system  : system prompt text
    user    : user-turn content (may include conversation history + instructions)
    timeout : seconds before raising asyncio.TimeoutError (default: config.LLM_TIMEOUT)

    Returns
    -------
    str  – raw LLM output (may need parse_json_response() to interpret)

    Raises
    ------
    RuntimeError – both providers failed
    """
    t = timeout or config.LLM_TIMEOUT
    primary   = config.LLM_PROVIDER.lower()
    primary_fn   = _call_groq   if primary == "groq"   else _call_gemini
    fallback_fn  = _call_gemini if primary == "groq"   else _call_groq

    primary_err = None

    try:
        result = await primary_fn(system, user, t)
        logger.debug(
        "LLM call succeeded via %s (%d chars)",
        primary,
        len(result),
        )
        return result

    except Exception as e:
        primary_err = e
        logger.warning(
            "Primary LLM (%s) failed: %s — trying fallback.",
            primary,
            e,
        )

    try:
        fallback_name = "gemini" if primary == "groq" else "groq"
        result = await fallback_fn(system, user, t)
        logger.info("LLM fallback (%s) succeeded.", fallback_name)
        return result
    except Exception as fallback_err:
        raise RuntimeError(
            f"Both LLM providers failed.\n"
            f"  Primary  ({primary}): {primary_err}\n"
            f"  Fallback ({fallback_name}): {fallback_err}"
        ) from fallback_err
