"""Domain objects shared by the conversational search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Constraint:
    """One preference extracted from the current shopping conversation."""

    attribute: str
    value: str
    strength: str
    confidence: float
    source_turn: int
    status: str = "active"

    @property
    def is_active(self) -> bool:
        """Return whether the constraint should affect retrieval and ranking."""

        return self.status == "active"


@dataclass
class SessionState:
    """Compact, mutable memory for one evaluator conversation."""

    profile_text: str
    route: str = "unknown"
    category: str = ""
    constraints: list[Constraint] = field(default_factory=list)
    asked_attributes: set[str] = field(default_factory=set)
    seen_attributes: set[str] = field(default_factory=set)
    unrestricted_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None

    def active_constraints(self) -> list[Constraint]:
        """Return only preferences that have not been overridden."""

        return [constraint for constraint in self.constraints if constraint.is_active]

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint unless the same active value is already present."""

        normalized_value = constraint.value.casefold().strip()
        duplicate = any(
            existing.is_active
            and existing.attribute == constraint.attribute
            and existing.value.casefold().strip() == normalized_value
            for existing in self.constraints
        )
        if not duplicate:
            self.constraints.append(constraint)

    def override_preferences(self) -> None:
        """Deactivate prior session preferences after an explicit intent change."""

        for constraint in self.constraints:
            if constraint.is_active:
                constraint.status = "overridden"
        self.seen_attributes.clear()


@dataclass(frozen=True)
class RetrievalCandidate:
    """A catalog item produced by first-stage multi-route retrieval."""

    parent_asin: str
    fusion_score: float
    best_retrieval_rank: int


@dataclass(frozen=True)
class ProductDocument:
    """Normalized catalog fields used by the second-stage reranker."""

    parent_asin: str
    title: str
    categories: str
    features: str
    details: str
    store: str
    description: str
    price: float | None
    all_text: str


@dataclass(frozen=True)
class RankedCandidate:
    """A retrieval candidate with its second-stage compatibility score."""

    parent_asin: str
    final_score: float
    retrieval_score: float

