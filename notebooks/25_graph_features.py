"""Phase 4 Day 25: inspect graph-based skill signals."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import build_skill_graph, load_phase3_context
from src.utils.paths import DOCS_DIR


def main() -> None:
    try:
        context = load_phase3_context(require_labels=False)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nCopy the released jobs/candidates files into data/raw and rerun."
        ) from exc

    graph = build_skill_graph(context)
    central = sorted(graph.skill_freq.items(), key=lambda item: item[1], reverse=True)[:20]

    lines = [
        "# Skill Graph Features\n\n",
        "| Skill | Frequency | Centrality |\n",
        "|---|---:|---:|\n",
    ]
    for skill, freq in central:
        lines.append(f"| {skill} | {freq} | {graph.skill_centrality(skill):.3f} |\n")

    example_score = graph.candidate_graph_score(
        candidate_skills=["Python", "SQL", "pandas"],
        required_skills=["Python", "Machine Learning", "TensorFlow"],
    )
    lines.extend(
        [
            "\n## Example\n\n",
            f"- Candidate graph score for `[Python, SQL, pandas]` vs `[Python, Machine Learning, TensorFlow]`: `{example_score:.3f}`\n",
        ]
    )

    output_path = DOCS_DIR / "graph_features.md"
    output_path.write_text("".join(lines), encoding="utf-8")
    print(f"Saved graph-feature summary to: {output_path}")


if __name__ == "__main__":
    main()
