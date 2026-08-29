"""Second-stage ranking based on explicit conversational compatibility."""

from __future__ import annotations

from starter.models import (
    Constraint,
    ProductDocument,
    RankedCandidate,
    RetrievalCandidate,
    SessionState,
)
from starter.retrieval import CatalogSearch
from starter.text import normalize_text, token_coverage


class ConstraintReranker:
    """Reorder retrieved products using category and active slot constraints."""

    def __init__(self, catalog: CatalogSearch) -> None:
        self.catalog = catalog

    @staticmethod
    def _attribute_text(product: ProductDocument, attribute: str) -> str:
        """Select the catalog fields most informative for an attribute."""

        if attribute == "brand":
            return f"{product.store} {product.title}"
        if attribute == "budget":
            return product.all_text
        if attribute in {"material", "color", "size", "style"}:
            return " ".join(
                [product.title, product.features, product.details, product.description]
            )
        if attribute in {"feature", "use_case"}:
            return " ".join(
                [product.title, product.features, product.description, product.details]
            )
        return product.all_text

    def _constraint_score(
        self,
        product: ProductDocument,
        constraint: Constraint,
    ) -> float:
        """Score one product against one active hard or soft preference."""

        normalized_value = normalize_text(constraint.value)
        exact_match = bool(normalized_value) and normalized_value in product.all_text
        overall_coverage = token_coverage(constraint.value, product.all_text)
        field_coverage = token_coverage(
            constraint.value,
            self._attribute_text(product, constraint.attribute),
        )
        confidence = constraint.confidence

        if exact_match:
            score = (10.0 if constraint.strength == "hard" else 6.0) * confidence
        else:
            coverage = 0.7 * overall_coverage + 0.3 * field_coverage
            score = (7.0 if constraint.strength == "hard" else 4.0) * coverage * confidence

        # Field agreement distinguishes a genuine material/brand/etc. match
        # from a coincidental mention elsewhere in a long description.
        score += 1.5 * field_coverage * confidence

        if constraint.strength == "hard" and overall_coverage < 0.35:
            score -= 6.0 * confidence
        return score

    def rank(
        self,
        candidates: list[RetrievalCandidate],
        state: SessionState,
        limit: int,
    ) -> list[RankedCandidate]:
        """Return the best candidates after explicit compatibility scoring."""

        if not candidates:
            return []
        maximum_fusion_score = max(candidate.fusion_score for candidate in candidates)
        ranked: list[RankedCandidate] = []

        for candidate in candidates:
            product = self.catalog.products[candidate.parent_asin]
            normalized_retrieval = candidate.fusion_score / maximum_fusion_score
            score = 6.0 * normalized_retrieval
            score += 2.0 / (1.0 + (candidate.best_retrieval_rank - 1) / 20.0)

            if state.category:
                category_coverage = token_coverage(state.category, product.categories)
                score += 4.0 * category_coverage

            for constraint in state.active_constraints():
                score += self._constraint_score(product, constraint)

            ranked.append(
                RankedCandidate(
                    parent_asin=candidate.parent_asin,
                    final_score=score,
                    retrieval_score=candidate.fusion_score,
                )
            )

        ranked.sort(
            key=lambda item: (-item.final_score, -item.retrieval_score, item.parent_asin)
        )
        return ranked[:limit]
