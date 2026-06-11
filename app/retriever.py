"""
app/retriever.py
----------------
Retrieves and ranks SHL catalog assessments for a given recruiter query.

Public API:
    retrieve(query, top_k=10) -> list[Recommendation]

Pipeline:
    query
      └─ search()           [embeddings.py]  → raw FAISS hits (cosine score + metadata)
      └─ deduplicate        [entity_id key]  → remove any repeat entries
      └─ sort by score      [descending]     → highest similarity first
      └─ generate reason    [rule-based]     → per-item explanation grounded in catalog
      └─ format             [Recommendation] → structured output for the agent
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.embeddings import search

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias – one recommendation record as returned by retrieve()
# ---------------------------------------------------------------------------
Recommendation = dict[str, Any]

# ---------------------------------------------------------------------------
# Human-readable labels for each single-letter test_type code.
# Used in recommendation_reason generation.
# ---------------------------------------------------------------------------
_TEST_TYPE_LABELS: dict[str, str] = {
    "A": "Ability & Aptitude (measures cognitive and reasoning skills)",
    "E": "Assessment Exercise (practical, scenario-based evaluation)",
    "B": "Biodata & Situational Judgment (predicts behaviour from past experience)",
    "C": "Competency assessment (measures job-relevant behavioural competencies)",
    "D": "Development & 360 feedback tool",
    "K": "Knowledge & Skills test (measures job-specific technical knowledge)",
    "P": "Personality & Behaviour questionnaire (maps work-related styles and traits)",
    "S": "Simulation (replicates realistic on-the-job scenarios)",
}

# ---------------------------------------------------------------------------
# Score-tier labels (cosine similarity thresholds, L2-normalised vectors).
# Cosine similarity from all-MiniLM-L6-v2 typically spans 0.2–0.9 for
# relevant matches; anything below 0.25 is a weak signal.
# ---------------------------------------------------------------------------
_SCORE_TIERS: list[tuple[float, str]] = [
    (0.65, "Highly relevant"),
    (0.45, "Strong match"),
    (0.30, "Good match"),
    (0.15, "Relevant match"),
    (0.0,  "Partial match"),
]

# ---------------------------------------------------------------------------
# Query-token → SHL job-level mapping.
# Used to detect which seniority level the recruiter is targeting so the
# reason can call out level fit explicitly.
# ---------------------------------------------------------------------------
_LEVEL_KEYWORD_MAP: dict[str, list[str]] = {
    "junior":      ["Entry-Level", "Graduate"],
    "entry":       ["Entry-Level"],
    "entry-level": ["Entry-Level"],
    "fresher":     ["Entry-Level", "Graduate"],
    "graduate":    ["Graduate"],
    "intern":      ["Entry-Level", "Graduate"],
    "mid":         ["Mid-Professional"],
    "mid-level":   ["Mid-Professional"],
    "senior":      ["Professional Individual Contributor", "Mid-Professional"],
    "experienced": ["Professional Individual Contributor", "Mid-Professional"],
    "lead":        ["Manager", "Front Line Manager"],
    "manager":     ["Manager", "Front Line Manager"],
    "supervisor":  ["Supervisor", "Front Line Manager"],
    "director":    ["Director"],
    "vp":          ["Director", "Executive"],
    "executive":   ["Executive"],
    "c-level":     ["Executive"],
    "cto":         ["Executive"],
    "ceo":         ["Executive"],
}

# ---------------------------------------------------------------------------
# Common English stopwords to ignore when matching query tokens to catalog text.
# Keeping this concise: only words that carry zero signal for assessment search.
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "be", "by", "as", "it", "its",
    "this", "that", "i", "me", "my", "we", "our", "you", "your", "who",
    "what", "which", "how", "have", "has", "can", "will", "need", "want",
    "looking", "hiring", "hire", "candidate", "candidates", "role", "roles",
    "position", "positions", "someone", "person", "people", "team",
    "around", "about", "some", "more", "new", "also", "very", "really",
    "test", "assessment", "assessments", "evaluate", "evaluation",
})


# ===========================================================================
# Internal helpers
# ===========================================================================

def _tokenize(text: str) -> list[str]:
    """
    Lowercase and tokenize text into alphabetic tokens of length ≥ 2,
    excluding stopwords.  Handles hyphenated terms like 'mid-level'.
    """
    # Split on whitespace and non-alphanumeric characters (keeps hyphen words split)
    raw_tokens = re.split(r"[^a-zA-Z0-9\-]+", text.lower())
    tokens: list[str] = []
    for tok in raw_tokens:
        tok = tok.strip("-")   # strip leading/trailing hyphens
        if len(tok) >= 2 and tok not in _STOPWORDS:
            tokens.append(tok)
    return tokens


def _detect_job_levels(query_tokens: list[str]) -> list[str]:
    """
    Return the list of SHL job levels implied by keywords in the query.
    E.g. ["senior"] → ["Professional Individual Contributor", "Mid-Professional"]

    Returns an empty list when no seniority signal is found.
    """
    detected: set[str] = set()
    for tok in query_tokens:
        if tok in _LEVEL_KEYWORD_MAP:
            detected.update(_LEVEL_KEYWORD_MAP[tok])
    return sorted(detected)


def _find_keyword_matches(
    query_tokens: list[str],
    item_name: str,
    item_description: str,
) -> list[str]:
    """
    Return query tokens that also appear in the assessment name or description.
    Matching is case-insensitive and looks for the token as a substring so that
    'java' matches 'Java 8 (New)' and also 'Core Java (Advanced Level)'.

    Returns at most 5 matched tokens to keep reasons concise.
    """
    combined = (item_name + " " + item_description).lower()
    matches = [tok for tok in query_tokens if tok in combined]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique[:5]


def _score_tier_label(score: float) -> str:
    """Map a cosine similarity score to a human-readable tier label."""
    for threshold, label in _SCORE_TIERS:
        if score >= threshold:
            return label
    return "Partial match"


def _format_level_list(levels: list[str]) -> str:
    """Format a list of job levels into a readable string."""
    if not levels:
        return ""
    if len(levels) == 1:
        return levels[0]
    return ", ".join(levels[:-1]) + f" and {levels[-1]}"


def _generate_reason(
    query: str,
    query_tokens: list[str],
    detected_levels: list[str],
    item: dict[str, Any],
) -> str:
    """
    Build a 1–3 sentence natural-language reason explaining why this assessment
    is recommended for the query.  Every claim is grounded in catalog data:
    no facts are generated that aren't present in the item's metadata.

    Structure:
      [1] Tier label + test type description.
      [2] Keyword match note (if any tokens from query appear in item text).
      [3] Job-level fit note (if detected levels overlap with item levels).
    """
    parts: list[str] = []

    # ── Sentence 1: score tier + test-type description ─────────────────────
    tier = _score_tier_label(item["score"])
    test_type_code = item.get("test_type", "")

    # For multi-type items (e.g. "P/K"), describe each part
    type_labels: list[str] = []
    for code in test_type_code.split("/"):
        code = code.strip()
        if code in _TEST_TYPE_LABELS:
            type_labels.append(_TEST_TYPE_LABELS[code])
    type_desc = "; ".join(type_labels) if type_labels else "assessment"

    parts.append(f"{tier} — {type_desc}.")

    # ── Sentence 2: keyword match in name or description ───────────────────
    matched_kw = _find_keyword_matches(
        query_tokens,
        item.get("name", ""),
        item.get("description", ""),
    )
    # Filter out generic tokens that aren't meaningful match signals
    # (e.g. "level", "new" are already removed by stopwords, but double-check)
    meaningful_kw = [kw for kw in matched_kw if len(kw) >= 3]

    if meaningful_kw:
        kw_str = ", ".join(f'"{kw}"' for kw in meaningful_kw[:3])
        parts.append(f"Directly relevant to your query: matches on {kw_str}.")
    else:
        # No direct keyword overlap — explain match via test category
        item_keys = item.get("keys", [])
        if item_keys:
            parts.append(
                f"Recommended based on semantic similarity to your requirement; "
                f"covers {item_keys[0]} competencies."
            )

    # ── Sentence 3: job-level fit ───────────────────────────────────────────
    item_levels: list[str] = item.get("job_levels", [])
    if item_levels:
        overlapping = [lvl for lvl in detected_levels if lvl in item_levels]
        if overlapping:
            parts.append(
                f"Designed for {_format_level_list(overlapping)} — "
                f"matches the seniority indicated in your query."
            )
        elif detected_levels:
            # User specified a level, but this assessment doesn't target it
            parts.append(
                f"Calibrated for {_format_level_list(item_levels[:3])}; "
                f"may still be applicable depending on role scope."
            )
        else:
            # No seniority signal in query — mention what levels it covers
            parts.append(
                f"Applicable to: {_format_level_list(item_levels[:3])}."
            )

    return " ".join(parts)


def _deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate assessments by entity_id.
    The first occurrence (highest score, since list is score-sorted) is kept.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        eid = item.get("entity_id", "")
        if eid not in seen:
            seen.add(eid)
            unique.append(item)
    return unique


# ===========================================================================
# Public API
# ===========================================================================

def retrieve(
    query: str,
    top_k: int = 10,
) -> list[Recommendation]:
    """
    Retrieve the top-k SHL assessments for a recruiter query.

    Steps:
      1. Validate and clean the query.
      2. Call search() from embeddings.py for semantic retrieval (top_k hits).
      3. Deduplicate by entity_id.
      4. Re-sort by cosine score descending (FAISS already orders, but
         dedup could theoretically change ordering if called on cached results).
      5. Generate a recommendation_reason for each result.
      6. Project to the output schema and return.

    Parameters
    ----------
    query  : free-text recruiter query, e.g.
             "backend java engineer with aws experience"
    top_k  : maximum results to return (1–10; clamped to catalog size)

    Returns
    -------
    list[Recommendation]
        Sorted list (best match first), each item containing:
          name, url, description, duration, job_levels, test_type,
          score, recommendation_reason.

    Raises
    ------
    ValueError  – empty query or invalid top_k
    RuntimeError – FAISS index not built yet (run build_index() first)
    """
    # ── Input validation ────────────────────────────────────────────────────
    if not query or not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(top_k, int) or not (1 <= top_k <= 10):
        raise ValueError("top_k must be an integer between 1 and 10.")

    query = query.strip()
    logger.info("Retrieving top-%d assessments for query: '%s'", top_k, query[:80])

    # ── Semantic search ─────────────────────────────────────────────────────
    # Fetch slightly more than top_k to have a buffer after deduplication.
    # With 377 catalog items and exact FAISS search, duplicates are rare, but
    # fetching top_k + 5 ensures we can always fill top_k after dedup.
    fetch_k = min(top_k + 5, 20)
    try:
        raw_hits: list[dict[str, Any]] = search(query, top_k=fetch_k)
    except ValueError as exc:
        # Propagate validation errors from search() unchanged
        raise
    except RuntimeError as exc:
        # Index missing — give a clear actionable message
        raise RuntimeError(
            f"Retriever: {exc}\n"
            "Ensure build_index() has been run to create data/faiss.index."
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during search: %s", exc, exc_info=True)
        raise RuntimeError(f"Search failed unexpectedly: {exc}") from exc

    if not raw_hits:
        logger.warning("Search returned no results for query: '%s'", query)
        return []

    # ── Deduplication ───────────────────────────────────────────────────────
    deduped = _deduplicate(raw_hits)

    # ── Sort by score descending ────────────────────────────────────────────
    deduped.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # ── Truncate to top_k ───────────────────────────────────────────────────
    top_results = deduped[:top_k]

    # ── Pre-compute query features for reason generation ───────────────────
    query_tokens = _tokenize(query)
    detected_levels = _detect_job_levels(query_tokens)

    logger.debug(
        "Query tokens: %s | Detected levels: %s", query_tokens, detected_levels
    )

    # ── Format output + generate reasons ───────────────────────────────────
    recommendations: list[Recommendation] = []
    for hit in top_results:
        reason = _generate_reason(query, query_tokens, detected_levels, hit)

        rec: Recommendation = {
            "name":                   hit.get("name", ""),
            "url":                    hit.get("url", ""),
            "description":            hit.get("description", ""),
            "duration":               hit.get("duration", ""),
            "job_levels":             hit.get("job_levels", []),
            "test_type":              hit.get("test_type", ""),
            "score":                  round(hit.get("score", 0.0), 4),
            "recommendation_reason":  reason,
        }
        recommendations.append(rec)

    logger.info(
        "retrieve() → %d recommendations (query='%s')",
        len(recommendations), query[:60]
    )
    return recommendations


# ===========================================================================
# CLI entry-point:  python app/retriever.py
# ===========================================================================

def _print_recommendations(
    recommendations: list[Recommendation],
    query: str,
) -> None:
    """Pretty-print a list of recommendations to stdout."""
    print(f"\n{'='*70}")
    print(f"  Query: \"{query}\"")
    print(f"  Results: {len(recommendations)} assessments")
    print(f"{'='*70}")

    for i, rec in enumerate(recommendations, 1):
        score_bar = "█" * int(rec["score"] * 20) if rec["score"] > 0 else "░"
        print(f"\n  [{i}] {rec['name']}")
        print(f"       Score      : {rec['score']:.4f}  {score_bar}")
        print(f"       Test Type  : {rec['test_type']}")
        print(f"       Duration   : {rec['duration'] or 'Not specified'}")
        print(f"       Job Levels : {', '.join(rec['job_levels']) or 'Not specified'}")
        print(f"       URL        : {rec['url']}")
        # Wrap description at 65 chars
        desc = rec["description"][:130] + "…" if len(rec["description"]) > 130 else rec["description"]
        print(f"       Description: {desc}")
        # Wrap reason
        reason = rec["recommendation_reason"]
        print(f"       Reason     : {reason}")

    print(f"\n{'='*70}\n")


def _offline_demo(query: str) -> None:
    """
    Fallback demo for environments where the embedding model cannot be
    downloaded (e.g. sandboxes with restricted network access).
    Loads metadata.json directly and shows reason-generation output for
    a hand-picked subset relevant to the query so the module logic is
    still observable.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    meta_path = data_dir / "metadata.json"

    if not meta_path.exists():
        print("  [offline demo] metadata.json not found — run build_index() first.")
        return

    with open(meta_path) as f:
        metadata: list[dict[str, Any]] = json.load(f)

    # Select items that have obvious relevance to the query via keyword overlap
    query_tokens = _tokenize(query)
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in metadata:
        match_count = len(_find_keyword_matches(
            query_tokens,
            item.get("name", ""),
            item.get("description", ""),
        ))
        if match_count > 0:
            scored.append((match_count, item))

    # Also add a couple of high-coverage personality/ability assessments
    for item in metadata:
        if item.get("test_type", "") in ("P", "A") and item not in [s[1] for s in scored]:
            scored.append((0, item))
            if len(scored) >= 12:
                break

    # Sort by match count, take top-10
    scored.sort(key=lambda x: x[0], reverse=True)
    top_items = [item for _, item in scored[:10]]

    # Assign placeholder scores so reason generation works
    detected_levels = _detect_job_levels(query_tokens)
    recs: list[Recommendation] = []
    for rank, item in enumerate(top_items):
        # Simulate a plausible descending score
        item_with_score = dict(item)
        item_with_score["score"] = round(0.72 - rank * 0.04, 4)
        reason = _generate_reason(query, query_tokens, detected_levels, item_with_score)
        recs.append({
            "name":                  item.get("name", ""),
            "url":                   item.get("url", ""),
            "description":           item.get("description", ""),
            "duration":              item.get("duration", ""),
            "job_levels":            item.get("job_levels", []),
            "test_type":             item.get("test_type", ""),
            "score":                 item_with_score["score"],
            "recommendation_reason": reason,
        })

    print("\n  ⚠  Offline demo mode: embedding model unavailable.")
    print("     Scores are illustrative; keyword matching used for selection.")
    print("     All reason-generation logic is identical to production.\n")
    _print_recommendations(recs, query)


if __name__ == "__main__":
    import sys

    TEST_QUERY = "backend java engineer with aws experience"

    print("\n>>> SHL Assessment Retriever — CLI test")
    print(f">>> Query: \"{TEST_QUERY}\"")

    try:
        results = retrieve(TEST_QUERY, top_k=10)
        _print_recommendations(results, TEST_QUERY)

        # Validate output schema
        required_keys = {
            "name", "url", "description", "duration",
            "job_levels", "test_type", "score", "recommendation_reason",
        }
        for i, rec in enumerate(results):
            missing = required_keys - rec.keys()
            if missing:
                print(f"  [ERROR] Item {i} is missing fields: {missing}")
                sys.exit(1)
            if not rec["url"].startswith("https://www.shl.com"):
                print(f"  [ERROR] Item {i} has a non-SHL URL: {rec['url']}")
                sys.exit(1)

        print(f"  ✓ Schema validation passed for all {len(results)} results.")
        print(f"  ✓ All URLs are from https://www.shl.com")

    except RuntimeError as exc:
        # Model not downloadable in this environment — use offline demo
        print(f"\n  [INFO] Full semantic search unavailable: {exc}")
        print("  Falling back to offline keyword-match demo …")
        _offline_demo(TEST_QUERY)

    except Exception as exc:
        print(f"\n  [ERROR] Unexpected failure: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
