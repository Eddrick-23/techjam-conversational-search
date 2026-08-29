"""Public Agent entry point for the conversational product-search pipeline."""

from __future__ import annotations

from pathlib import Path

from starter.conversation import ConversationInterpreter
from starter.models import SessionState
from starter.ranking import ConstraintReranker
from starter.retrieval import CatalogSearch, MultiRouteRetriever
from starter.text import flatten_text


QUESTION_ORDER = (
    "material", "feature", "color", "style", "size",
    "use_case", "budget", "brand", "other",
)
QUESTION_TEXT = {
    "material": "Do you have a material preference?",
    "feature": "Which feature or product detail matters most to you?",
    "color": "Do you have a color preference?",
    "style": "What style or fit would you prefer?",
    "size": "Do you have any size or width requirements?",
    "use_case": "What activity or situation will you use it for?",
    "budget": "Do you have a target budget?",
    "brand": "Do you have a preferred brand?",
    "other": "Is there one other requirement that would help narrow the options?",
}


class Agent:
    """Coordinate conversation state, broad retrieval, and precise reranking.

    The first stage retrieves a generous product pool from several field-weighted
    BM25 routes. The second stage inspects those candidates against structured,
    active conversation constraints before returning the final Top K.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog = CatalogSearch(self.catalog_path)
        self.retriever = MultiRouteRetriever(self.catalog)
        self.reranker = ConstraintReranker(self.catalog)
        self.interpreter = ConversationInterpreter()
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Create isolated state for a new shopping conversation."""

        profile_parts = [
            flatten_text(user_profile.get("preference_tags")),
            flatten_text(user_profile.get("summary")),
        ]
        self._sessions[session_id] = SessionState(
            profile_text=" ".join(part for part in profile_parts if part)
        )

    @staticmethod
    def _choose_question(state: SessionState) -> str | None:
        """Choose the next unanswered, non-repeating clarification slot."""

        for attribute in QUESTION_ORDER:
            if attribute in state.asked_attributes:
                continue
            if attribute in state.unrestricted_attributes:
                continue
            if attribute in state.seen_attributes and attribute != "other":
                continue
            state.asked_attributes.add(attribute)
            state.last_asked_attribute = attribute
            return attribute
        state.last_asked_attribute = None
        return None

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        """Process one turn and return a question plus reranked recommendations."""

        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        state = self._sessions[session_id]
        current_fact = self.interpreter.update(state, user_message, turn)

        # Retrieval favors recall; reranking then favors constraint compatibility.
        candidates = self.retriever.retrieve(state, current_fact, limit=300)
        ranked = self.reranker.rank(candidates, state, limit=top_k)
        recommendations = [
            {"parent_asin": candidate.parent_asin}
            for candidate in ranked
        ]

        ask_attribute = self._choose_question(state)
        message = (
            QUESTION_TEXT[ask_attribute]
            if ask_attribute
            else "These are the strongest matches for the preferences you've shared."
        )
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
