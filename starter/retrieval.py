"""Catalog indexing and first-stage multi-route BM25 retrieval."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from starter.models import ProductDocument, RetrievalCandidate, SessionState
from starter.text import flatten_text, normalize_text, terms


class CatalogSearch:
    """Own the in-memory FTS index and normalized product documents."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.products: dict[str, ProductDocument] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                raw_fields = {
                    "title": flatten_text(product.get("title")),
                    "categories": flatten_text(product.get("categories")),
                    "features": flatten_text(product.get("features")),
                    "details": flatten_text(product.get("details")),
                    "store": flatten_text(product.get("store")),
                    "description": flatten_text(product.get("description")),
                }
                price_value = product.get("price")
                price = float(price_value) if isinstance(price_value, (int, float)) else None
                normalized = {
                    name: normalize_text(value)
                    for name, value in raw_fields.items()
                }
                price_text = f"price {price} budget around {price}" if price is not None else ""
                all_text = " ".join([*normalized.values(), price_text]).strip()
                self.products[parent_asin] = ProductDocument(
                    parent_asin=parent_asin,
                    title=normalized["title"],
                    categories=normalized["categories"],
                    features=normalized["features"],
                    details=normalized["details"],
                    store=normalized["store"],
                    description=normalized["description"],
                    price=price,
                    all_text=all_text,
                )
                batch.append(
                    (
                        parent_asin,
                        raw_fields["title"],
                        raw_fields["categories"],
                        raw_fields["features"],
                        raw_fields["details"],
                        raw_fields["store"],
                        raw_fields["description"],
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def search(self, query: str, limit: int) -> list[tuple[str, int]]:
        """Return product IDs and ranks for one field-weighted BM25 query."""

        unique_terms = list(dict.fromkeys(terms(query)))[:80]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 7.0, 5.0, 3.0, 2.5, 2.0, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [(str(row[0]), rank) for rank, row in enumerate(rows, start=1)]


class MultiRouteRetriever:
    """Fuse category, conversation, current-turn, and profile searches."""

    def __init__(self, catalog: CatalogSearch) -> None:
        self.catalog = catalog

    def retrieve(
        self,
        state: SessionState,
        current_fact: str,
        limit: int = 300,
    ) -> list[RetrievalCandidate]:
        """Generate a broad candidate pool using reciprocal-rank fusion."""

        active_values = [constraint.value for constraint in state.active_constraints()]
        accumulated = " ".join([state.category, *active_values]).strip()
        routes: list[tuple[str, float, int]] = []
        if accumulated:
            routes.append((accumulated, 1.5, 300))
        if state.category:
            routes.append((state.category, 1.25, 180))
        if current_fact:
            routes.append((current_fact, 1.1, 220))
        for value in active_values[-4:]:
            routes.append((value, 0.8, 120))
        if state.profile_text:
            routes.append((state.profile_text, 0.12, 60))

        fused_scores: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        for query, weight, route_limit in routes:
            for parent_asin, rank in self.catalog.search(query, route_limit):
                fused_scores[parent_asin] = (
                    fused_scores.get(parent_asin, 0.0) + weight / (40.0 + rank)
                )
                best_rank[parent_asin] = min(best_rank.get(parent_asin, rank), rank)

        ordered = sorted(
            fused_scores,
            key=lambda asin: (-fused_scores[asin], best_rank[asin], asin),
        )
        return [
            RetrievalCandidate(
                parent_asin=parent_asin,
                fusion_score=fused_scores[parent_asin],
                best_retrieval_rank=best_rank[parent_asin],
            )
            for parent_asin in ordered[:limit]
        ]
