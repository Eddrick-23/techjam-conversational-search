# Starter Agent Architecture

`Agent.respond(...)` is intentionally a thin orchestration method. One turn flows through four stages:

```text
customer message
    -> ConversationInterpreter.update
    -> MultiRouteRetriever.retrieve (up to 300 candidates)
    -> ConstraintReranker.rank (final Top K)
    -> clarification question + recommendations
```

## Modules

- `agent.py`: evaluator-facing `Agent` and turn orchestration.
- `conversation.py`: category, route, constraint, boundary, and override interpretation.
- `retrieval.py`: in-memory SQLite FTS5 catalog and reciprocal-rank-fused BM25 routes.
- `ranking.py`: second-stage category and hard/soft constraint compatibility scoring.
- `models.py`: shared session, constraint, product, and candidate data objects.
- `text.py`: catalog flattening, normalization, tokenization, and coverage helpers.

The retrieval stage is deliberately recall-oriented. It finds a broad candidate pool without deciding the final recommendations. The reranking stage then inspects those candidates against the active structured conversation state and returns the final ordered list.

No LLM or network connection is required, so reported token usage remains zero.
