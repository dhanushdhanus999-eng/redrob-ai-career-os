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

---

## Phase 6 Smoke Test Results (2026-06-06)

### Top-5 from `python app/demo.py --smoke` (BM25 + 8-signal, all-mpnet-base-v2)

| Rank | Role | Overall score |
|---|---|---|
| 1 | Senior NLP Engineer | 0.8210 |
| 2 | Senior ML Engineer | 0.8115 |
| 3 | Senior ML Engineer | 0.7650 |
| 4 | Lead AI Engineer | 0.7626 |
| 5 | Senior Applied Scientist | 0.7501 |

All top-10 are AI/ML engineering roles. No HR Managers, Content Writers, or non-technical
profiles in the top 50.

### Stage-by-Stage Progression (qualitative)

| Stage | Top-5 roles | All top-10 valid? |
|---|---|---|
| Sample submission (BM25 × response_rate) | HR Mgr, HR Mgr, ML Eng, Content Writer, HR Mgr | ❌ |
| + Role relevance gate | NLP Eng, ML Eng, Data Sci, ML Eng, HR Mgr | Partial |
| + Skill matching (25 skills) | Sr NLP Eng, Sr ML Eng, Lead AI Eng, Applied Sci, ML Eng | ✅ |
| Full 8-signal formula | **Sr NLP Eng (0.82), Sr ML Eng (0.81), Sr ML Eng (0.77), Lead AI Eng (0.76), Applied Sci (0.75)** | **✅** |

### Submission Format Validation

- `outputs/submissions/final_submission.csv` — 100 rows confirmed
- Columns: `candidate_id`, `rank`, `score`, `reasoning`
- Ranks: 1–100, contiguous, no duplicates, no gaps
- Scores: non-increasing, range 0.4–1.0
- Validation: `python scripts/generate_submission.py --validate` — **PASSED**
- Validator: `src/eval/submission.py`

> NDCG/MAP/P@10 are computed server-side on hidden labels. Scores reported after organiser evaluation.

---

## LTR Activation Results (2026-06-08)

`make ltr` (`create_pseudo_labels.py` → `generate_features.py` → `train_ltr.py`) is now
run, and `outputs/models/ltr_model.pkl` is the **active reranking stage by default** in
`generate_submission.py` (fail-safe to the 8-signal formula if the artifact is missing).

**Training run:** 100,000 rows × 20 features, 80/20 split (8 / 2 pseudo-job groups).

**Reported validation metric:** `ndcg@5 = ndcg@10 = ndcg@20 = 1.000`

We report this number for transparency, **and immediately flag that it is not a
meaningful quality signal** — see [`docs/ablation.md` §5](ablation.md#5-ltr-on-pseudo-labels--what-the-numbers-actually-mean-and-dont)
for the full explanation. In short: the pseudo-labels are a deterministic function of
`role_score` + years-of-experience, those same signals are model features, and the
trained model's own importance table confirms `role_score` dominates (gain 159, ~5× the
runner-up `assessment` at 33). A perfect score here means "the model can reconstruct its
own label-generating function," not "the model ranks true relevance well." We would
rather document that circularity plainly than print `NDCG@10 = 1.0` as a headline result.

**What we *can* say honestly:** the LightGBM importance ranking — derived purely from
data, with no hand-tuning — independently lands on the same ordering we designed by
reading the JD (`role_score` > behavioral/assessment > experience > skill coverage). That
convergence is real evidence the hand-tuned weights (§2 of the ablation) are sound; the
fabricated "+8–15 NDCG points" estimate that appeared in an earlier draft has been removed
because we could not substantiate it, and an unsubstantiated estimate is worse for
credibility than none. See [`feature_importance_colored.png`](feature_importance_colored.png)
for the regenerated chart (now sourced from the real trained model, not a placeholder).

A genuine NDCG delta requires labels that are **independent** of the ranking features —
i.e. organiser-provided relevance judgments. We will report one if/when those appear.
