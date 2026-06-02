"""Skill co-occurrence graph features for job-candidate ranking."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict


class SkillGraph:
    """Skill co-occurrence graph built from parsed job requirements."""

    def __init__(self) -> None:
        self.co_occurrence: Dict[tuple[str, str], int] = defaultdict(int)
        self.skill_freq: Dict[str, int] = defaultdict(int)

    def fit(self, all_skill_lists: list[list[str]]) -> None:
        """Build graph statistics from job skill lists."""
        self.co_occurrence.clear()
        self.skill_freq.clear()

        for skills in all_skill_lists:
            unique_skills = list({skill for skill in skills if skill})
            for skill in unique_skills:
                self.skill_freq[skill] += 1
            for left, right in combinations(sorted(unique_skills), 2):
                self.co_occurrence[(left, right)] += 1

    def skill_centrality(self, skill: str) -> float:
        """Return how broadly a skill co-occurs with other skills."""
        if not skill:
            return 0.0
        total_co = sum(
            weight
            for (left, right), weight in self.co_occurrence.items()
            if left == skill or right == skill
        )
        frequency = self.skill_freq.get(skill, 0)
        return float(total_co / (frequency + 1))

    def candidate_graph_score(
        self,
        candidate_skills: list[str],
        required_skills: list[str],
    ) -> float:
        """Score how well candidate skills cover the required skill neighborhood."""
        if not required_skills or not candidate_skills:
            return 0.0

        required_neighbors = set()
        for required_skill in required_skills:
            for (left, right), count in self.co_occurrence.items():
                if count < 3:
                    continue
                if left == required_skill:
                    required_neighbors.add(right)
                elif right == required_skill:
                    required_neighbors.add(left)

        if not required_neighbors:
            return 0.0

        covered = len(required_neighbors.intersection(set(candidate_skills)))
        return float(covered / len(required_neighbors))
