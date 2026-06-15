"""
app/agent.py
------------
Core agent that powers the POST /chat endpoint.

Every call to run_agent() is fully stateless — the entire conversation
history arrives in the request and no per-session state is stored server-side.

Decision pipeline (one HTTP request = one pass through this pipeline):
  1.  Count turns → check if turn cap forces a final recommendation.
  2.  Build a single LLM prompt containing the full conversation history.
  3.  Call the LLM → parse structured JSON (intent + reply + derived_query).
  4.  Route:
        clarify   → return reply, empty recommendations
        off_topic → return polite refusal, empty recommendations
        compare   → look up both items in catalog, call LLM for comparison
        recommend / refine → call retrieve(), format ChatRecommendation list
        force_rec → retrieve() with last-user-message as query fallback
  5.  Return ChatResponse.

Hallucination prevention:
  - The reply text never names specific assessments (those come from retrieval).
  - Comparison uses actual catalog metadata injected into the LLM prompt.
  - All URLs go through ChatRecommendation.url_must_be_shl validator.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import config
from app.chat_schema import ChatRequest, ChatResponse, ChatRecommendation, Message
from app.llm import call_llm, parse_json_response
from app.prompts import (
    AGENT_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_USER_TEMPLATE,
    FALLBACK_CLARIFYING_QUESTIONS,
    FALLBACK_OFF_TOPIC_REPLY,
)
from app.retriever import retrieve

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level metadata cache  (loaded once, kept in memory)
# =============================================================================

_metadata: list[dict[str, Any]] = []


def _load_metadata() -> list[dict[str, Any]]:
    """Load catalog metadata from disk (lazy, cached after first call)."""
    global _metadata
    if _metadata:
        return _metadata
    path = config.METADATA_PATH
    if path.exists():
        with path.open() as fh:
            _metadata = json.load(fh)
        logger.info("Metadata loaded: %d items", len(_metadata))
    else:
        logger.warning("Metadata file not found at %s", path)
    return _metadata


# =============================================================================
# Catalog lookup helpers
# =============================================================================

def _find_catalog_item(name: str) -> dict[str, Any] | None:
    """
    Locate a catalog item by name.  Tries exact match first, then
    case-insensitive substring match so 'OPQ32r' finds 'Occupational Personality
    Questionnaire OPQ32r'.
    """
    meta = _load_metadata()
    name_lower = name.strip().lower()
    for item in meta:
        if item.get("name", "").strip().lower() == name_lower:
            return item
    for item in meta:
        if name_lower in item.get("name", "").lower():
            return item
    return None


def _format_item_for_comparison(item: dict[str, Any]) -> str:
    """Format a catalog item's key facts for injection into the comparison prompt."""
    return "\n".join([
        f"Type        : {item.get('test_type', '?')}  ({', '.join(item.get('keys', []))})",
        f"Description : {(item.get('description', '') or 'Not available')[:400]}",
        f"Duration    : {item.get('duration', 'Not specified') or 'Not specified'}",
        f"Job Levels  : {', '.join(item.get('job_levels', [])) or 'Not specified'}",
        f"URL         : {item.get('url', '')}",
    ])


# =============================================================================
# Conversation helpers
# =============================================================================

def _count_turns(messages: list[Message]) -> int:
    """Total message count (user + assistant combined)."""
    return len(messages)


def _last_user_message(messages: list[Message]) -> str:
    """Return the content of the last user-role message."""
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _build_conversation_block(messages: list[Message]) -> str:
    """Serialise the message history for embedding in the LLM prompt."""
    lines: list[str] = []
    for m in messages:
        label = "Recruiter" if m.role == "user" else "Assistant"
        lines.append(f"{label}: {m.content}")
    return "\n".join(lines)


# =============================================================================
# Output helpers
# =============================================================================

def _to_chat_recs(raw: list[dict[str, Any]]) -> list[ChatRecommendation]:
    """
    Convert retriever output dicts to validated ChatRecommendation objects.
    Invalid items (e.g. non-SHL URLs that slipped through) are logged and skipped.
    """
    result: list[ChatRecommendation] = []
    for item in raw[:config.MAX_RECOMMENDATIONS]:
        try:
            result.append(ChatRecommendation(
                name=item["name"],
                url=item["url"],
                test_type=item.get("test_type", ""),
            ))
        except Exception as exc:
            logger.warning("Skipping invalid recommendation '%s': %s", item.get("name"), exc)
    return result


# =============================================================================
# Fallback when LLM is unavailable
# =============================================================================

def _fallback_response(messages: list[Message], force_recommend: bool) -> ChatResponse:
    """
    Deterministic fallback used when both LLM providers fail.

    If force_recommend is True (turn cap approaching) we do a keyword-based
    retrieval so the evaluator still gets a shortlist.  Otherwise we return
    the first canned clarifying question.
    """
    if force_recommend:
        query = _last_user_message(messages) or "general SHL assessment"
        try:
            raw  = retrieve(query, top_k=5)
            recs = _to_chat_recs(raw)
            return ChatResponse(
                reply="Based on your request, here are some relevant assessments.",
                recommendations=recs,
            )
        except Exception as exc:
            logger.error("Fallback retrieval also failed: %s", exc)

    return ChatResponse(
        reply=FALLBACK_CLARIFYING_QUESTIONS[0],
        recommendations=[],
    )


# =============================================================================
# Comparison handler
# =============================================================================

async def _handle_compare(
    compare_targets: list[str],
    messages: list[Message],
) -> ChatResponse:
    """
    Look up both assessments in the catalog and generate a grounded comparison.
    A second LLM call is made with the actual catalog data injected — this is
    the only reliable way to prevent hallucination of assessment features.
    """
    if len(compare_targets) < 2:
        return ChatResponse(
            reply=(
                "To compare assessments, please name both of them. "
                "For example: 'Compare the OPQ32r and the Verify Numerical Reasoning test.'"
            ),
            recommendations=[],
        )

    item1 = _find_catalog_item(compare_targets[0])
    item2 = _find_catalog_item(compare_targets[1])

    missing = [t for t, i in zip(compare_targets, [item1, item2]) if i is None]
    if missing:
        return ChatResponse(
            reply=(
                f"I couldn't find {', '.join(missing)!r} in the SHL catalog. "
                "Please check the name and try again."
            ),
            recommendations=[],
        )

    user_prompt = COMPARISON_USER_TEMPLATE.format(
        user_question=_last_user_message(messages),
        name1=item1["name"],         # type: ignore[index]
        data1=_format_item_for_comparison(item1),  # type: ignore[arg-type]
        name2=item2["name"],         # type: ignore[index]
        data2=_format_item_for_comparison(item2),  # type: ignore[arg-type]
    )

    try:
        reply_text = await call_llm(COMPARISON_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        logger.error("Comparison LLM call failed: %s", exc)
        reply_text = (
            f"Both {item1['name']} and {item2['name']} are in the SHL catalog. "  # type: ignore[index]
            "I'm unable to generate a detailed comparison right now, "
            "but you can view their catalog pages via the URLs below."
        )

    return ChatResponse(reply=reply_text.strip(), recommendations=[], end_of_conversation=False)


# =============================================================================
# Deterministic role-extraction  (safety net for LLM mis-classification)
# =============================================================================
# The LLM reliably detects roles stated in natural language ("Java developer")
# because those formats appear in the prompt examples.  It can fail on
# structured key-value input ("Role: Java Developer. Experience: 2 years.")
# because that format has no matching example in the clarify/recommend rules.
#
# This regex covers the structured case only — natural-language mentions are
# correctly handled by the LLM and do not need a client-side override.
# =============================================================================

_STRUCTURED_ROLE = re.compile(
    r"(?:role|job\s*title|position|job)\s*[:\-]\s*(.+?)(?:\s*[.,;\n]|$)",
    re.IGNORECASE | re.MULTILINE,
)

# Words that can follow "Role:" etc. but are NOT job titles
_NON_ROLE_WORDS: frozenset[str] = frozenset({
    "assessment", "assessments", "test", "tests", "tool", "tools",
    "solution", "solutions", "resource", "help", "something", "platform",
})


def _extract_role_hint(text: str) -> str | None:
    """
    Lightweight deterministic check: does *text* name a role in structured
    key-value format ("Role: X", "Job: X", "Position: X", "Job Title: X")?

    Returns the extracted role string if found, otherwise None.

    Used exclusively as a safety net in run_agent() to override an LLM
    "clarify" intent when the structured format has already provided a role.
    Natural-language role mentions ("sales manager", "nurse") are intentionally
    left for the LLM to handle — they are covered by the prompt examples.
    """
    m = _STRUCTURED_ROLE.search(text)
    if m:
        role = m.group(1).strip().rstrip(".,;")
        first_word = role.lower().split()[0] if role else ""
        if role and first_word not in _NON_ROLE_WORDS:
            return role
    return None




async def run_agent(request: ChatRequest) -> ChatResponse:
    """
    Process one /chat request and return the next agent reply.

    Parameters
    ----------
    request : ChatRequest containing the full conversation history.

    Returns
    -------
    ChatResponse with reply, recommendations (possibly empty), and
    end_of_conversation flag.
    """
    messages      = request.messages
    total_turns   = _count_turns(messages)
    last_user_msg = _last_user_message(messages)

    # ── Turn-cap check ────────────────────────────────────────────────────────
    # When total_turns >= MAX_TURNS - 1 we are at or past the last valid turn.
    # The agent MUST produce a recommendation — asking another question would
    # exceed the evaluator's 8-turn cap and the conversation would be cut off.
    force_recommend = total_turns >= config.MAX_TURNS - 1

    # ── Build LLM prompt ──────────────────────────────────────────────────────
    system_prompt = AGENT_SYSTEM_PROMPT.format(
        force_at_turn=config.MAX_TURNS - 1
    )
    user_prompt = (
        f"[Turn {total_turns} of {config.MAX_TURNS} max"
        + (" — FORCE RECOMMENDATION NOW]" if force_recommend else "]")
        + f"\n\nConversation so far:\n{_build_conversation_block(messages)}"
        + "\n\nAnalyse the conversation and respond with the JSON object."
    )

    # ── Primary LLM call ─────────────────────────────────────────────────────
    try:
        raw_text = await call_llm(system_prompt, user_prompt)
        print("\n===== RAW LLM OUTPUT =====")
        print(raw_text)
        print("==========================\n")
        parsed   = parse_json_response(raw_text)
    except Exception as exc:
        logger.error("Agent LLM call/parse failed: %s", exc)
        return _fallback_response(messages, force_recommend)

    # ── Extract structured fields ──────────────────────────────────────────────
    intent          = str(parsed.get("intent",          "clarify"))
    reply_text      = str(parsed.get("reply",           "")).strip()
    derived_query   = str(parsed.get("derived_query",   "")).strip()
    compare_targets = list(parsed.get("compare_targets", []))
    end_of_conv     = bool(parsed.get("end_of_conversation", False))

    # Guard: if no reply was generated, use a safe default
    if not reply_text:
        reply_text = "Could you tell me more about the role you're hiring for?"

    # ── Turn-cap override ─────────────────────────────────────────────────────
    # If the LLM still chose to clarify despite the force flag, override it.
    if force_recommend and intent not in ("recommend", "refine", "compare", "off_topic"):
        logger.info("Turn cap override: forcing recommend intent.")
        intent = "recommend"
        if not derived_query:
            derived_query = last_user_msg

    # ── Route on intent ───────────────────────────────────────────────────────

    if intent == "off_topic":
        return ChatResponse(reply=reply_text, recommendations=[], end_of_conversation=False)

    if intent == "clarify":
        # Safety-net: if the user already stated a role in structured format
        # ("Role: Java Developer"), the LLM mis-classified — override to recommend
        # so retrieval runs immediately instead of asking a redundant question.
        role_hint = _extract_role_hint(last_user_msg)
        if role_hint:
            logger.info(
                "Clarify-override: structured role '%s' detected in user message "
                "— switching intent to recommend.",
                role_hint,
            )
            intent = "recommend"
            if not derived_query:
                # Use the full message as the retrieval query; the retriever
                # will extract the salient terms (role, skills, experience).
                derived_query = last_user_msg
        else:
            return ChatResponse(reply=reply_text, recommendations=[], end_of_conversation=False)

    if intent == "compare":
        return await _handle_compare(compare_targets, messages)

    if intent in ("recommend", "refine"):
        # Guard: derived_query should be non-empty for retrieval
        if not derived_query:
            logger.warning("LLM returned recommend intent but empty derived_query.")
            return ChatResponse(
                reply="Could you share a bit more about the role? "
                      "For example, the job title and seniority level.",
                recommendations=[],
            )

        try:
            raw_results = retrieve(derived_query, top_k=config.MAX_RECOMMENDATIONS)
            recs        = _to_chat_recs(raw_results)
        except Exception as exc:
            logger.error("Retrieval failed for query '%s': %s", derived_query[:80], exc)
            recs = []

        # If retrieval returned nothing, treat as still-clarifying
        if not recs:
            return ChatResponse(
                reply=(
                    reply_text + " (I wasn't able to retrieve specific assessments — "
                    "could you add more detail about the role?)"
                ),
                recommendations=[],
            )

        return ChatResponse(
            reply=reply_text,
            recommendations=recs,
            end_of_conversation=end_of_conv,
        )

    # Unknown intent — default to clarify
    logger.warning("Unexpected intent '%s' from LLM — defaulting to clarify.", intent)
    return ChatResponse(reply=reply_text, recommendations=[], end_of_conversation=False)
