# Intelligent Candidate Discovery - India Runs Track 1

An end-to-end AI ranking pipeline for Redrob AI's India Runs Track 1 challenge.
It goes beyond title and keyword matching by combining structured job understanding,
hybrid BM25 + dense retrieval, 30+ fit features, behavioral activity signals,
reranking stages, and per-candidate natural-language explanations.

**Live Demo:** [Deploy to Hugging Face Spaces — see `docs/HUGGINGFACE_DEPLOY.md`]  
**Video:** [Record with Loom/YouTube — script at `docs/DEMO_VIDEO_SCRIPT.md`]  
**Dataset:** Official India Runs public bundle, normalised into `data/processed/`  
**Challenge shape:** One Senior AI Engineer JD · 100,000 candidates · hidden evaluation

---

## Reproduce in 3 commands

```bash
git clone <repo-url> && cd india-runs-track1
pip install -e ".[dev]"
make reproduce
```

Or step by step:

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
python scripts/build_indices.py          # build BM25 index (~3 min first run)
python scripts/generate_submission.py --validate   # generate + validate 100-row CSV
```

The submission is written to `outputs/submissions/final_submission.csv`.

---

## Results

The official scoring formula: `0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10`.

The public challenge bundle contains **no relevance labels**. NDCG, MAP and precision
are computed server-side on the organiser's hidden test set. All pipeline stages are
fully implemented and validated for schema correctness locally.

| Pipeline stage | Implementation status | Key artifact |
|---|---|---|
| BM25 recall | Complete | `src/retrieval/bm25_retriever.py` |
| Dense retrieval (BGE-large) | Complete | `src/retrieval/dense_retriever.py` |
| Hybrid recall (BM25 + dense, RRF) | Complete | `src/retrieval/hybrid_retriever.py` |
| Skill matching (exact + family + fuzzy) | Complete | `src/utils/skill_ontology.py` |
| Experience and seniority features | Complete | `src/features/experience_features.py` |
| Behavioral signals (8 Redrob signals) | Complete | `src/features/behavioral_features.py` |
| Semantic features (multi-model) | Complete | `src/features/semantic_features.py` |
| LightGBM LambdaRank | Ready — awaits organiser labels | `src/models/ltr_model.py` |
| Cross-encoder reranking | Ready — awaits final artifacts | `src/ranking/cross_encoder.py` |
| LLM listwise reranking (Claude Haiku) | Ready — `--llm-rerank` flag | `src/ranking/llm_reranker.py` |
| Submission generation | Complete | `scripts/generate_submission.py` |
| Submission validation | Complete | `python -m src.eval.validate_submission --pred <file>` |

> Metric columns remain blank — fabricating NDCG numbers without labels
> is a disqualifying antipattern. Scores will be reported after organiser
> evaluation or official feedback.

---

## Architecture

![System architecture](docs/architecture_diagram.png)

```
Job Description
  -> Stage 0: Structured parsing (skills, seniority, domain, years, location)
  -> Stage 1: Hybrid recall — BM25 + BGE-large dense retrieval + RRF fusion
  -> Stage 2: Feature engineering (30+ features per pair)
       Semantic similarity (multi-model cosine)
       Skill overlap (exact + ontology family + fuzzy)
       Experience and seniority alignment
       ★ Behavioral signals (recency, availability, engagement, assessments)
  -> Stage 3: LightGBM LambdaRank  (supervised — activates with labels)
  -> Stage 4: Cross-encoder reranking  (BGE-reranker)
  -> Stage 5: LLM listwise reranking  (Claude Haiku, cached)
  -> Ranked shortlist + per-candidate reasoning
```

---

## What Makes This Different

1. **Behavioral signals are first-class.** Recency, open-to-work state,
   recruiter response rate, profile completeness, saved-by-recruiter counts,
   GitHub activity, and assessment quality are modeled beside text relevance.
   The brief explicitly calls these out; most submissions will ignore them.
2. **The pipeline is staged.** Fast BM25 recall narrows 100K to ~1K;
   multi-signal scoring narrows to 100; rerankers sharpen precision.
3. **Every output is explainable.** Each ranked candidate gets a rationale
   that cites matched skills, experience fit, seniority alignment, and
   activity confidence.
4. **Honest about what it can't measure.** The repo validates locally
   but does not invent NDCG numbers without ground-truth labels.
5. **Fully reproducible.** `make reproduce` runs end-to-end in a clean clone.
   A `Dockerfile` is provided for environment isolation.

---

## Visual Evidence

![Feature signal map](docs/feature_importance_colored.png)

![NDCG progression status](docs/ndcg_progression.png)

Both visuals are label-aware. When a trained LTR model artifact is present,
`python docs/architecture.py` replaces the signal map with real feature
importance and the progression chart with real NDCG values.

---

## Running the Demo

```powershell
# Quick smoke test (no Gradio required)
python app/demo.py --smoke

# Interactive Gradio UI
pip install -r app/requirements.txt
python app/demo.py
```

The demo ranks the full 100K-candidate pool using BM25 recall plus
transparent skill, experience, and behavioral scoring. The first run
builds a BM25 cache under `outputs/models/` (~3 min).

---

## Repository Structure

```
scripts/             submission generator, index builder, data parser
src/
  data/              challenge bundle loader, schema inference
  retrieval/         BM25, dense, hybrid retrievers
  features/          semantic, skill, experience, behavioral, graph features
  models/            LightGBM LambdaRank wrapper, fine-tune scaffold
  ranking/           cross-encoder, LLM reranker, explainer
  eval/              NDCG/MAP/precision metrics, submission validator
  utils/             skill ontology, caching, text utils, fast index
app/                 Gradio demo and Hugging Face Spaces entry point
configs/             best_hparams.json for LTR training
notebooks/           one script per phase day (EDA through Phase 4)
docs/                Blueprint, status reports, video script, visuals
tests/               pytest coverage (metrics, parsing, retrieval, features)
outputs/             runtime — models, cache, submissions (git-ignored)
```

---

## Tech Stack

| Layer | Choice |
|---|---|
| Retrieval | `rank-bm25`, `sentence-transformers` (BGE-large-en-v1.5), FAISS HNSW |
| Features | `rapidfuzz`, `scikit-learn`, custom skill ontology |
| Ranker | LightGBM LambdaRank (`lightgbm`) |
| Reranking | BGE-reranker cross-encoder, Claude Haiku LLM listwise |
| Demo | Gradio 5, Plotly |
| Evaluation | NDCG / MAP / precision, submission contract validator |
| Reproducibility | Makefile, Dockerfile, pinned `requirements.txt` |

---

## Judge Documents

- [System Blueprint](docs/BLUEPRINT.md) — detailed stage-by-stage design decisions
- [Hugging Face Deploy Notes](docs/HUGGINGFACE_DEPLOY.md) — live Space setup
- [Demo Video Script](docs/DEMO_VIDEO_SCRIPT.md) — 5–7 min walkthrough script
- [Phase 5 Status](docs/phase5_status.md) — demo and narrative completion log

## Video Walkthrough

Recording script: `docs/DEMO_VIDEO_SCRIPT.md`  
Live video: *(paste Loom / YouTube URL here after recording)*

---

## Author

Built for India Runs Hackathon 2026 · Track 1, Data & AI Challenge by Redrob AI.
