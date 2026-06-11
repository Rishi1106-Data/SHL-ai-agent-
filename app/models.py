"""
app/models.py
-------------
Pydantic v2 data models for the SHL Assessment Recommender.

Hierarchy
---------
Recommendation           – one assessed assessment item from the SHL catalog
RecommendationResponse   – the full payload returned by the /chat endpoint
                           when the agent has committed to a shortlist

These models serve three purposes:
  1. Runtime validation — Pydantic rejects bad data before it leaves the service.
  2. Serialization — .model_dump() / .model_dump_json() produce the exact JSON
     shape the assignment evaluator expects.
  3. Documentation — FastAPI auto-generates OpenAPI schema from these classes.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# ===========================================================================
# Recommendation
# ===========================================================================

class Recommendation(BaseModel):
    """
    A single SHL assessment recommendation.

    Every field maps directly to a column in the catalog / a key produced by
    retriever.retrieve().  The schema must stay in sync with the evaluator's
    expectations — do not rename or remove fields.
    """

    name: Annotated[str, Field(
        description="Display name of the assessment exactly as it appears in the SHL catalog.",
        min_length=1,
        examples=["Java 8 (New)", "Occupational Personality Questionnaire OPQ32r"],
    )]

    url: Annotated[str, Field(
        description=(
            "Canonical SHL catalog URL for this assessment.  "
            "Must begin with https://www.shl.com — no other domains are permitted."
        ),
        examples=["https://www.shl.com/products/product-catalog/view/java-8-new/"],
    )]

    description: Annotated[str, Field(
        description="Full assessment description scraped from the SHL catalog.",
        default="",
    )]

    duration: Annotated[str, Field(
        description=(
            "Approximate completion time as a human-readable string, "
            "e.g. '30 minutes'.  Empty string when not specified in the catalog."
        ),
        default="",
        examples=["30 minutes", "17 minutes", ""],
    )]

    job_levels: Annotated[list[str], Field(
        description=(
            "SHL job levels this assessment is designed for, "
            "e.g. ['Mid-Professional', 'Professional Individual Contributor']."
        ),
        default_factory=list,
        examples=[["Mid-Professional", "Professional Individual Contributor"]],
    )]

    test_type: Annotated[str, Field(
        description=(
            "Single-letter code(s) indicating the assessment category. "
            "Slash-separated for multi-category items. "
            "A=Ability & Aptitude, B=Biodata & Situational Judgment, "
            "C=Competencies, D=Development & 360, E=Assessment Exercises, "
            "K=Knowledge & Skills, P=Personality & Behavior, S=Simulations."
        ),
        default="",
        examples=["K", "P", "A", "P/K", "S"],
    )]

    score: Annotated[float, Field(
        description=(
            "Cosine similarity score from the FAISS index (L2-normalised vectors). "
            "Range [-1, 1]; higher is more semantically similar to the query."
        ),
        ge=-1.0,
        le=1.0,
        examples=[0.82, 0.61, 0.44],
    )]

    recommendation_reason: Annotated[str, Field(
        description=(
            "Human-readable explanation of why this assessment was recommended "
            "for the query.  Grounded solely in catalog data — no hallucinated claims."
        ),
        min_length=1,
        examples=[
            "Highly relevant — Knowledge & Skills test. "
            "Directly relevant to your query: matches on \"java\". "
            "Applicable to: Mid-Professional and Professional Individual Contributor."
        ],
    )]

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("url")
    @classmethod
    def url_must_be_shl(cls, v: str) -> str:
        """
        Reject any URL that does not point to the SHL domain.
        This is a hard guardrail — the evaluator checks that every URL
        in recommendations comes from the scraped catalog.
        """
        if not v.startswith("https://www.shl.com"):
            raise ValueError(
                f"Recommendation URL must start with 'https://www.shl.com', got: {v!r}"
            )
        return v

    @field_validator("score", mode="before")
    @classmethod
    def score_must_be_finite(cls, v: object) -> object:
        """
        Guard against NaN / Inf leaking in from FAISS on degenerate vectors.
        Runs in 'before' mode so it fires before Pydantic's ge/le range checks,
        giving a clear 'finite float' error rather than a confusing
        'less_than_equal' error for NaN.
        """
        import math
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"score must be a finite float, got {v!r}")
        # Trim floating-point noise after type coercion (done post-validation)
        return v

    @field_validator("score", mode="after")
    @classmethod
    def score_trim_noise(cls, v: float) -> float:
        """Trim floating-point noise once the value has passed range validation."""
        return round(v, 6)

    @field_validator("job_levels")
    @classmethod
    def job_levels_no_duplicates(cls, v: list[str]) -> list[str]:
        """Remove duplicates while preserving insertion order."""
        seen: set[str] = set()
        return [lvl for lvl in v if not (lvl in seen or seen.add(lvl))]  # type: ignore[func-returns-value]

    model_config = {
        # Silently strip extra fields from retriever output (e.g. entity_id)
        # so the API response never leaks internal keys.
        "extra": "ignore",
        # Use enum values, not enum names, in serialisation (future-proofing).
        "use_enum_values": True,
    }


# ===========================================================================
# RecommendationResponse
# ===========================================================================

class RecommendationResponse(BaseModel):
    """
    The complete recommendation payload returned by POST /chat when the agent
    has committed to a shortlist.

    When the agent is still gathering context, recommendations is an empty
    list and this model is still used — just with count=0 and recommendations=[].
    """

    query: Annotated[str, Field(
        description=(
            "The recruiter's distilled intent as understood by the agent "
            "at the time of retrieval.  May differ from the raw user message "
            "if the agent refined or expanded the query across turns."
        ),
        min_length=1,
        examples=["mid-level Java developer with stakeholder management skills"],
    )]

    count: Annotated[int, Field(
        description=(
            "Number of recommendations in this response.  "
            "Always equals len(recommendations) — auto-corrected by the model validator "
            "if the caller passes a wrong value.  "
            "0 when the agent is still clarifying; 1–10 when a shortlist is committed."
        ),
        ge=0,
        # No le=10: the real cap is enforced by max_length=10 on recommendations.
        # Removing it lets the model_validator auto-correct wrong counts before
        # the final object is built, rather than rejecting them early.
        examples=[5, 3, 0],
    )]

    recommendations: Annotated[list[Recommendation], Field(
        description=(
            "Ordered list of recommended SHL assessments (best match first). "
            "Empty list when the agent is gathering context or refusing an off-topic query. "
            "1–10 items when the agent has committed to a shortlist."
        ),
        default_factory=list,
        min_length=0,
        max_length=10,
    )]

    # ------------------------------------------------------------------
    # Cross-field validator
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def count_matches_list_length(self) -> "RecommendationResponse":
        """
        Ensure count always equals len(recommendations).
        Callers should set count=len(recommendations), but this validator
        auto-corrects mismatches rather than rejecting valid data.
        """
        if self.count != len(self.recommendations):
            # Auto-correct: derive count from the list rather than raising.
            # This makes the model resilient to agent code that forgets to
            # update count when it trims the shortlist.
            object.__setattr__(self, "count", len(self.recommendations))
        return self

    model_config = {
        "extra": "ignore",
    }

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, query: str) -> "RecommendationResponse":
        """
        Return a response with no recommendations.
        Use this while the agent is still gathering context.

        Example
        -------
        >>> RecommendationResponse.empty("I need an assessment")
        RecommendationResponse(query='I need an assessment', count=0, recommendations=[])
        """
        return cls(query=query, count=0, recommendations=[])

    @classmethod
    def from_retriever(
        cls,
        query: str,
        raw: list[dict],
    ) -> "RecommendationResponse":
        """
        Build a RecommendationResponse directly from the list of dicts
        returned by retriever.retrieve().

        Parameters
        ----------
        query : the recruiter's distilled query string
        raw   : output of retriever.retrieve() — list of recommendation dicts

        Returns
        -------
        RecommendationResponse with validated Recommendation objects.

        Raises
        ------
        pydantic.ValidationError  if any item fails field validation
                                  (e.g. a non-SHL URL slipped through).
        """
        recs = [Recommendation(**item) for item in raw]
        return cls(query=query, count=len(recs), recommendations=recs)
