from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.models import Constraint, RetrievalCandidate, SessionState


class AgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "COTTON",
                "title": "Women's cotton walking shirt",
                "categories": ["Women", "Shirts"],
                "features": ["soft cotton", "lightweight"],
                "details": {"Department": "Women"},
                "store": "Example",
                "description": ["comfortable walking top"],
            },
            {
                "parent_asin": "WOOL",
                "title": "Women's wool walking shirt",
                "categories": ["Women", "Shirts"],
                "features": ["warm merino wool", "winter weight"],
                "details": {"Department": "Women"},
                "store": "Example",
                "description": ["comfortable cold-weather walking top"],
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(catalog_path)
        self.agent.reset("session", {"preference_tags": [], "summary": ""})

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_accumulates_constraints_and_asks_a_new_attribute(self) -> None:
        first = self.agent.respond(
            "session",
            "I'm looking for women's shirts, but I'm still exploring.",
            1,
            10,
        )
        self.assertEqual(first["ask_attribute"], "material")

        second = self.agent.respond(
            "session",
            "For that, what matters is: wool.",
            2,
            10,
        )
        self.assertEqual(second["recommendations"][0]["parent_asin"], "WOOL")
        self.assertEqual(second["ask_attribute"], "feature")
        constraint = self.agent._sessions["session"].active_constraints()[0]
        self.assertEqual(constraint.attribute, "material")
        self.assertEqual(constraint.value, "wool")
        self.assertEqual(constraint.strength, "hard")

    def test_override_erases_the_old_preference(self) -> None:
        self.agent.respond(
            "session",
            "I'm looking for women's shirts. cotton is important.",
            1,
            10,
        )
        response = self.agent.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: wool.",
            2,
            10,
        )
        self.assertEqual(response["recommendations"][0]["parent_asin"], "WOOL")
        state = self.agent._sessions["session"]
        active_values = [constraint.value for constraint in state.active_constraints()]
        self.assertEqual(active_values, ["wool"])
        cotton = next(
            constraint for constraint in state.constraints
            if "cotton" in constraint.value.lower()
        )
        self.assertEqual(cotton.status, "overridden")

    def test_no_preference_reply_is_not_added_to_search(self) -> None:
        self.agent.respond(
            "session",
            "I'm looking for women's shirts, but I'm still exploring.",
            1,
            10,
        )
        self.agent.respond(
            "session",
            "I don't have a preference for material; please use your judgment.",
            2,
            10,
        )
        state = self.agent._sessions["session"]
        self.assertIn("material", state.unrestricted_attributes)
        self.assertFalse(
            any("judgment" in constraint.value for constraint in state.constraints)
        )

    def test_reranker_promotes_an_exact_hard_constraint_match(self) -> None:
        state = SessionState(profile_text="", category="women's shirts")
        state.add_constraint(
            Constraint(
                attribute="material",
                value="warm merino wool",
                strength="hard",
                confidence=1.0,
                source_turn=2,
            )
        )
        candidates = [
            RetrievalCandidate("COTTON", fusion_score=0.10, best_retrieval_rank=1),
            RetrievalCandidate("WOOL", fusion_score=0.05, best_retrieval_rank=2),
        ]

        ranked = self.agent.reranker.rank(candidates, state, limit=2)

        self.assertEqual(ranked[0].parent_asin, "WOOL")


if __name__ == "__main__":
    unittest.main()
