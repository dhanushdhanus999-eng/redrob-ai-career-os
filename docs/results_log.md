# Results Log

## Status

The public bundle is now present and Phase 1 has been executed on real data.
However, the challenge remains hidden-eval, so there are still no public
ranking metrics to report.

What is now validated locally:

- the bundle contains 100,000 candidates and one released JD
- the repo can materialize canonical processed datasets:
  - `data/processed/challenge_jobs.csv`
  - `data/processed/challenge_candidates.parquet`
- the sample submission passes the published validator rules
- any local CSV can now be checked with:
  `python -m src.eval.validate_submission --pred <file>`

## Implemented pipelines

- `notebooks/05_bm25_baseline.py`
- `notebooks/06_dense_baseline.py`
- `notebooks/07_hybrid_retrieval.py`
- `notebooks/08_jd_parsing.py`
- `notebooks/09_candidate_parsing.py`
- `notebooks/10_skill_ontology.py`
- `notebooks/11_semantic_features.py`
- `notebooks/12_skill_features.py`
- `notebooks/13_experience_features.py`
- `notebooks/14_behavioral_features.py`
- `notebooks/15_train_ltr.py`
- `notebooks/16_embedding_ensemble.py`
- `notebooks/17_cross_encoder.py`
- `notebooks/18_llm_reranker.py`
- `notebooks/19_explainability.py`
- `notebooks/20_phase3_pipeline.py`

## Planned entries

- BM25 ranking run against the released JD
- Dense embedding ranking run against the released JD
- Hybrid recall ranking run against the released JD
- Submission-quality comparisons using manual review and hidden-eval outcome
- LTR / reranking experiments if organizers ever expose labels later
