"""
app/chat_schema.py
------------------
Pydantic v2 models that define the POST /chat wire format.

Important: The /chat recommendation schema is intentionally minimal —
just name, url, test_type — as specified in the assignment.
The richer Recommendation model in models.py is for GET /recommend only.

The evaluator validates every response against this exact shape.
Do NOT add or rename fields.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """One turn in the conversation history."""
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    """
    Full conversation history sent on every POST /chat call.
    The API is stateless — the client owns the history.
    """
    messages: list[Message] = Field(
        min_length=1,
        max_length=20,   # safety cap; assignment max is 8 turns
        description="Full conversation history. Last message must be from the user.",
    )

    @field_validator("messages")
    @classmethod
    def last_message_is_user(cls, v: list[Message]) -> list[Message]:
        """
        The agent always responds to the user's latest message.
        Reject requests where the last message is from the assistant —
        that would mean the client is replaying an already-answered state.
        """
        if v and v[-1].role != "user":
            raise ValueError(
                "The last message must have role='user'. "
                "Include the full conversation history and end with the user's latest input."
            )
        return v


class ChatRecommendation(BaseModel):
    """
    Minimal assessment record returned inside POST /chat responses.
    Schema is non-negotiable per the assignment evaluator.
    """
    name:      str = Field(description="Assessment name exactly as in the SHL catalog.")
    url:       str = Field(description="Full SHL catalog URL for this assessment.")
    test_type: str = Field(default="", description="Single-letter type code(s), e.g. 'K', 'P/A'.")

    @field_validator("url")
    @classmethod
    def url_must_be_shl(cls, v: str) -> str:
        if not v.startswith("https://www.shl.com"):
            raise ValueError(
                f"Every recommendation URL must start with 'https://www.shl.com'. Got: {v!r}"
            )
        return v


class ChatResponse(BaseModel):
    """
    Response from POST /chat. Shape is fixed by the assignment spec:
      reply                 – conversational text from the agent
      recommendations       – empty while clarifying; 1-10 items when committed
      end_of_conversation   – true only when the agent considers the task done
    """
    reply:               str                     = Field(description="Agent's conversational reply.")
    recommendations:     list[ChatRecommendation] = Field(
        default_factory=list,
        max_length=10,
        description="Empty while gathering context; 1-10 catalog items when recommending.",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="Set to true only when the agent has fully satisfied the request.",
    )
