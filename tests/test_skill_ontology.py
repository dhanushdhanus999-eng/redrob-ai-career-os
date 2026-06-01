"""Unit tests for skill normalisation and matching."""

from __future__ import annotations

import unittest

from src.utils.skill_ontology import SkillMatcher, normalize_skill


class SkillOntologyTests(unittest.TestCase):
    def test_normalize_skill_maps_common_aliases(self) -> None:
        self.assertEqual(normalize_skill("reactjs"), "React")
        self.assertEqual(normalize_skill("py"), "Python")

    def test_skill_matcher_gives_family_credit(self) -> None:
        matcher = SkillMatcher()
        result = matcher.match_score(
            required_skills=["Python", "Machine Learning", "AWS"],
            candidate_skills=["py", "TensorFlow", "gcp"],
        )

        self.assertEqual(result["n_exact_matched"], 1)
        self.assertEqual(result["n_family_matched"], 2)
        self.assertEqual(result["n_missing"], 0)
        self.assertGreater(result["composite_score"], 0.6)


if __name__ == "__main__":
    unittest.main()
