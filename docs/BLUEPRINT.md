# System Blueprint - Intelligent Candidate Discovery

## 1. Problem Framing

The challenge is a single-query hidden-evaluation ranking problem: one released
Senior AI Engineer job description and 100,000 noisy candidate profiles. The
system must return exactly 100 candidates in rank order, with scores and
reasoning, while avoiding brittle title or keyword-only traps.

The official brief emphasizes contextual relevance and behavioral signals, so
the system treats profile text, structured fit, and activity/availability as
separate evidence streams. Because labels are hidden, local work focuses on
schema correctness, submission safety, transparent heuristics, and readiness for
supervised reranking when labels or leaderboard feedback appear.

## 2. Stage-by-Stage Design Decisions

### Stage 0: Why Structured Parsing?

The JD contains nuanced positive and negative requirements, not just a bag of
skills. Parsing extracts role title, seniority, years, technical themes,
location, and responsibilities so downstream features can reason about fit
instead of matching long prose blindly.

Candidate parsing similarly normalizes skills, seniority, experience, education,
location, and profile completeness. This keeps feature generation deterministic
and makes explanations reusable across demo, submission, and analysis flows.

### Stage 1: Why Hybrid Recall?

BM25 is strong for exact terms such as `Qdrant`, `BGE`, `NDCG`, and `LoRA`.
Dense retrieval is better for semantic neighbors such as ranking, recommender
systems, search relevance, and applied ML infrastructure.

The hybrid retriever fuses lexical and dense rankings with reciprocal rank
fusion. That keeps exact-match precision without missing candidates whose
profiles describe the same experience with different vocabulary.

### Stage 2: Why These Four Feature Groups?

Semantic features capture textual alignment between the JD and the full
candidate profile. Skill features separate exact, family, and fuzzy matches so
`PyTorch` and `Machine Learning` can receive graded credit rather than binary
failure.

Experience features score years, seniority, and education alignment. Behavioral
features use Redrob-specific signals such as last activity, open-to-work state,
profile completeness, response rate, search appearance, saved-by-recruiter
counts, and assessment quality.

### Stage 3: Why LightGBM over Neural Ranking?

The candidate pool has many heterogeneous tabular signals, and LightGBM is a
strong fit for sparse, mixed-scale ranking features. LambdaRank also provides
feature importance, fast CPU inference, and simpler failure analysis than an
opaque neural ranker.

The wrapper is implemented but training remains blocked locally by missing
labels. Once labels or trusted pseudo-labels exist, the model can consume the
merged Phase 3 feature frames directly.

### Stage 4: Why Cross-Encoder Reranking?

Cross-encoders are too expensive for 100,000 candidates but valuable on a short
recall set. The design reranks only the top candidates after retrieval and
feature ranking, where deeper query-document interaction can improve precision.

The repo keeps this stage optional so local smoke tests remain CPU-friendly.
Production or final-submission runs can enable it once model artifacts and
runtime budget are available.

### Stage 5: Why LLM Listwise Reranking?

The JD includes qualitative negative criteria such as title-chasing, consulting
only backgrounds, and weak external validation. An LLM listwise pass can reason
over the final shortlist when structured features are too blunt.

This stage is deliberately cached and optional. It should be used only for the
last 30 to 100 candidates so cost, latency, and reproducibility remain under
control.

## 3. Feature Engineering Details

### Semantic Features

Semantic features compare full JD text, responsibility text, and skill-focused
text against candidate profile text. The implementation supports multiple
embedding models and caches encoded batches so repeated experiments do not
recompute expensive vectors.

### Skill Features

Skill features use a small ontology with synonyms, canonical forms, family
matches, and fuzzy matching. This allows exact matches to score highest while
still giving partial credit for adjacent technologies in the same ecosystem.

### Experience Features

Experience features score total years against the JD's 5-9 year range, seniority
compatibility, education hints, and location alignment. Over-experience is not
treated as automatic failure because founding-team roles may value senior
system ownership.

### Behavioral Features

Behavioral features include last active recency, open-to-work status, profile
completeness, recruiter response rate, saved-by-recruiter counts, GitHub
activity, search appearances, and skill assessment quality. These signals are
especially important because the brief warns that intent and subtle activity
signals are lost in traditional filters.

## 4. Training Details

The training path builds job-candidate pairs, merges semantic, skill,
experience, behavioral, graph, and confidence features, then trains LightGBM
LambdaRank. Hyperparameter tuning and ablation scripts are present but write
blocked-status reports until organizer labels are available.

The expected supervised workflow is: create labels or import official labels,
build feature parquet files, train the LTR model, export `outputs/models/ltr_final`,
then regenerate the feature importance chart and final submission.

## 5. Evaluation Methodology

The repository implements NDCG, MAP, precision, and submission validation.
Because the public bundle has no relevance labels, local evaluation checks
schema correctness, duplicate prevention, score ordering, candidate membership,
and exact top-100 shape.

The hidden composite is `0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10`.
Metric tables in the README remain pending until official labels, leaderboard
feedback, or trusted validation judgments exist.

## 6. Limitations and Future Work

The largest limitation is the lack of local relevance labels, which blocks
honest NDCG reporting, ablation deltas, fairness conclusions, and final
hyperparameter selection. The current Gradio demo therefore uses transparent
BM25 plus feature heuristics rather than pretending to load a trained model.

Next steps are to deploy the Space, record the demo video, create a final
top-100 submission, and use any leaderboard or organizer feedback to calibrate
feature weights and supervised ranking artifacts.
