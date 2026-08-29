"""Interpret customer messages and update structured session memory."""

from __future__ import annotations

import re

from starter.models import Constraint, SessionState


MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)
NO_PREFERENCE_MARKERS = (
    "don't have a preference",
    "don't have an additional preference",
)
FACT_MARKERS = (
    "what i need is:",
    "what matters is:",
    "a key requirement is:",
)


def category_from_initial_message(message: str) -> str:
    """Extract the coarse catalog category disclosed in the first message."""

    match = re.search(r"\blooking for\s+(.+?)(?:[.,]|$)", message, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def classify_attribute(value: str) -> str:
    """Map a free-form requirement to one supported conversation attribute."""

    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under|around|less than)\s*\d", lowered):
        return "budget"
    if MATERIAL_RE.search(value):
        return "material"
    if COLOR_RE.search(value) or "color" in lowered:
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("style", "fit", "sleeve", "neckline", "department")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    if any(word in lowered for word in ("brand", "store", "manufacturer")):
        return "brand"
    return "feature"


def _constraint_text(message: str, turn: int) -> str:
    """Remove conversational boilerplate and retain newly disclosed facts."""

    lowered = message.lower()
    if any(marker in lowered for marker in NO_PREFERENCE_MARKERS):
        return ""
    if "options are not quite right" in lowered:
        return ""
    for marker in FACT_MARKERS:
        position = lowered.find(marker)
        if position >= 0:
            return message[position + len(marker):].strip(" .")

    if turn == 1:
        category_match = re.search(
            r"\blooking for\s+.+?(?:[.,]|$)",
            message,
            re.IGNORECASE,
        )
        if category_match:
            remainder = message[category_match.end():].strip(" ,.")
            if "still exploring" in remainder.lower():
                return ""
            return remainder
    return message.strip()


def _constraint_strength(message: str, turn: int) -> tuple[str, float]:
    """Infer how strongly a disclosed value should affect final ranking."""

    lowered = message.lower()
    if (
        "key requirement" in lowered
        or "what matters is" in lowered
        or "what i need is" in lowered
        or "must" in lowered
    ):
        return "hard", 0.95
    if turn == 1 and "still exploring" not in lowered:
        return "soft", 0.75
    return "soft", 0.70


class ConversationInterpreter:
    """Translate one customer turn into route, category, and slot updates."""

    def update(self, state: SessionState, message: str, turn: int) -> str:
        """Apply a message to session state and return its new retrieval text."""

        lowered = message.lower()
        if turn == 1:
            state.category = category_from_initial_message(message)
            state.route = "browsing" if "still exploring" in lowered else "buying"

        if state.last_asked_attribute and any(
            marker in lowered for marker in NO_PREFERENCE_MARKERS
        ):
            state.unrestricted_attributes.add(state.last_asked_attribute)

        if "actually, ignore my earlier preference" in lowered:
            state.override_preferences()
            state.route = "buying"

        fact_text = _constraint_text(message, turn)
        if not fact_text:
            return ""

        strength, confidence = _constraint_strength(message, turn)
        added_values: list[str] = []
        for raw_value in fact_text.split(";"):
            value = raw_value.strip(" ,.")
            if not value:
                continue
            attribute = classify_attribute(value)
            state.add_constraint(
                Constraint(
                    attribute=attribute,
                    value=value,
                    strength=strength,
                    confidence=confidence,
                    source_turn=turn,
                )
            )
            state.seen_attributes.add(attribute)
            added_values.append(value)
        return " ".join(added_values)
