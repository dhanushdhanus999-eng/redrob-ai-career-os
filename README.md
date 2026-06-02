# Redrob AI Career OS

Dataset-ready foundation for the Redrob AI Career OS candidate discovery and ranking platform.

## Current status

Phases 1 through 3 are implemented, and the released public Track 1 bundle is now
wired into the repository:

- repository scaffold and package layout
- pinned project configuration in `pyproject.toml`
- evaluation harness with ranking metrics and tests
- public-bundle discovery, DOCX extraction, and candidate flattening utilities
- canonical Phase 1 datasets generated from the released bundle:
  - `data/processed/challenge_jobs.csv`
  - `data/processed/challenge_candidates.parquet`
- EDA scripts for the released job description, candidate pool, and submission spec
- BM25, dense, and hybrid retrieval baselines
- structured job and candidate parsing utilities
- skill ontology and graded skill-matching utilities
- semantic, skill, experience, and behavioral feature engineering
- LightGBM LTR, cross-encoder reranking, LLM reranking, and explanations
- documentation, memory notes, and phase session logs

The released public bundle is a hidden-evaluation challenge:

- one released job description
- 100,000 released candidates
- no public labels or leaderboard
- a strict top-100 CSV submission contract

That means Phase 1 is fully runnable locally, while Phase 2 and Phase 3 can now
build rankings from real data but still cannot be locally scored against ground
truth until organizers reveal labels or final results.

## Quick start

1. Create a Python 3.11 virtual environment.
2. Install the project in editable mode:

   ```bash
   pip install -e ".[dev]"
   ```

3. Extract the official public bundle into `data/raw/india_runs_challenge/`.
4. Run the Phase 1 scripts:

   ```bash
   python notebooks/01_eda_jobs.py
   python notebooks/02_eda_candidates.py
   python notebooks/03_understand_labels.py
   python notebooks/04_create_splits_and_baseline.py
   ```

5. Validate any submission CSV locally before upload:

   ```bash
   python -m src.eval.validate_submission --pred outputs/submissions/your_submission.csv
   ```

6. Run the Phase 2 baselines and parsers once the canonical processed datasets exist:

   ```bash
   python notebooks/05_bm25_baseline.py
   python notebooks/06_dense_baseline.py
   python notebooks/07_hybrid_retrieval.py
   python notebooks/08_jd_parsing.py
   python notebooks/09_candidate_parsing.py
   python notebooks/10_skill_ontology.py
   ```

7. Run the Phase 3 core system once jobs and candidates are prepared:

   ```bash
   python notebooks/11_semantic_features.py
   python notebooks/12_skill_features.py
   python notebooks/13_experience_features.py
   python notebooks/14_behavioral_features.py
   python notebooks/15_train_ltr.py
   python notebooks/20_phase3_pipeline.py
   ```

8. Use `python -m src.eval.score_submission ...` only if labels become available later.

## Repository layout

```text
data/
  raw/ processed/ external/
notebooks/
src/
  data/ eval/ features/ models/ ranking/ utils/
configs/
outputs/
  submissions/ models/ logs/ cache/ figures/
tests/
docs/
Memory/
```

## Notes

- Challenge references are preserved under `Documents/`.
- Progress and reusable observations are tracked in `Memory/`.
- Dataset assumptions and blocked items are tracked in `docs/phase1_status.md`.
- Phase 2 implementation notes are tracked in `Documents/Phase 2/`.
- Phase 3 implementation notes are tracked in `Documents/Phase 3/`.
