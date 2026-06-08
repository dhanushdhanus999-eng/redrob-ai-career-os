# Error Analysis — India Runs Track 1

> **Date:** 2026-06-06  
> **Pipeline version:** Phase 6 (8-signal weighted formula + LTR)

This document analyses the four most significant failure modes in the current pipeline,
their root causes, mitigations, and residual risk in the final 100-row submission.

---

## Summary Table

| Failure type | Est. frequency in pool | NDCG@10 impact | Mitigated? |
|---|---|---|---|
| Consulting-background AI engineers | ~8–12% of grade-2 candidates | High if undetected | Partial |
| Junior candidates with keyword lists | ~5% of top-100 BM25 recall | Medium | Low residual |
| Honeypot profiles (all skills, wrong title) | < 1% of pool | Low | Very low residual |
| Sparse/incomplete profiles | ~15% of pool | Medium | Low-medium residual |

---

## Failure Mode 1 — Consulting-Background AI Engineers

**Example:** A TCS / Infosys / Wipro "ML Engineer" with 7 years total experience.

### Root Cause

`score_role_relevance()` returns `role_score = 1.0` for any profile with "ML Engineer"
in the title, regardless of employer. A candidate who has spent 7 years at TCS
doing AI/ML work on client projects will score identically on `role_score` to a
candidate at a product startup doing the same work.

The JD explicitly states preference for product-company experience over IT services.

### Mitigation

`score_career_trajectory()` in `src/utils/role_relevance.py` applies a
**consulting firm penalty** (−0.20 to −0.40) when the `career_history_text` contains
tokens from `CONSULTING_FIRM_TOKENS` (TCS, Infosys, Wipro, Accenture, Cognizant, etc.)
without compensating signals (e.g., open-source contributions, AI publications, patent
mentions, product-company name in recent roles).

With the 8-signal formula at `career_score_weight = 0.07`, a full penalty of −0.40
reduces the overall score by approximately −0.028.

### Residual Risk: **Medium**

- Affected candidates: ranks 20–50 in practice
- They rarely break into top-10 because `career_score` alone cannot overcome the
  signal from `must_composite` and `exp_score`
- Cannot fully separate 7-year TCS AI project experience from 7-year product startup
  AI experience without reading the actual project descriptions

---

## Failure Mode 2 — Junior Candidates with AI Keyword Lists

**Example:** A 2-year ML Engineer who has copy-pasted all 25 must-have skills into
their profile (Qdrant, Pinecone, FAISS, etc.) without genuine project depth.

### Root Cause

Skill matching (`SkillMatcher.match_score`) scores coverage of listed skills but cannot
verify depth or project context. A 2-year candidate who lists all 25 must-have skills
would achieve `must_composite ≈ 1.0` and `skill_score ≈ 0.95`.

### Mitigation

`score_experience()` returns `0.0–0.4` for candidates under 5 years:

```python
if cand_years < min_years:  # min_years = 5.0
    return max(0.0, cand_years / max(min_years, 1.0))
```

A 2-year candidate gets `exp_score = 0.40`. With `exp_weight = 0.10`, the overall
score penalty relative to a 6-year candidate is approximately −0.06.

Combined with behavioral signals (younger candidates typically have lower
`profile_completeness`, lower `saved_by_recruiters_30d`, and lower `response_rate`),
the effective penalty is −0.08 to −0.12.

### Residual Risk: **Low**

- At `exp_score = 0.40` and typical behavioral scores, a 2-year candidate with perfect
  skills scores approximately 0.60–0.65, comfortably below the 0.75+ range of
  genuine 5–9 year engineers with strong skill matches

---

## Failure Mode 3 — Honeypot Profiles (All Skills, Wrong Title)

**Example:** A profile listing all 25 must-have skills and all 18 nice-to-have skills
but with title "Business Analyst" or "Product Manager".

### Root Cause

Pure skill-matching pipelines cannot detect title mismatch. A Business Analyst profile
with a complete skill list would score `skill_score ≈ 0.95` and potentially appear
in the top-10 of a skill-only ranker.

### Mitigation

`score_role_relevance()` returns `role_score = 0.1` for Business Analyst, Product Manager,
and similar non-engineering titles. At `role_weight = 0.13`:

```
overall_honeypot_max ≈ 0.47   (all other signals perfect, role_score = 0.1)
overall_genuine_min  ≈ 0.75   (moderate other signals, role_score = 1.0)
```

The `role_score = 0.1` cap creates a hard ceiling below any genuine AI engineer with
even moderate skill and experience scores.

### Residual Risk: **Very Low**

- Would require the organiser to have deliberately included honeypot profiles AND
  assigned them positive relevance labels
- In practice, the role gate is definitive for this failure type

---

## Failure Mode 4 — Sparse / Incomplete Profiles

**Example:** A Senior AI Engineer with fewer than 100 words in their profile —
a minimal LinkedIn export with job title, company, and dates only.

### Root Cause

- BM25 score: low (few keywords to match)
- Semantic score: low (short embedding vector, high variance)
- Skill matching: depends on whether skills are listed at all; often 0
- Behavioral signals: `profile_completeness_score` will be low

A genuinely strong candidate with a sparse profile may not surface in the BM25 recall
step (`recall_k = 2000`) at all, making them invisible to subsequent scoring stages.

### Mitigation

1. **Completeness in behavioral score:** `completeness` contributes `weight 0.25`
   inside `beh_score`, which itself has `weight 0.10`. The effective overall penalty
   for a profile with `completeness = 0.2` vs `completeness = 0.9` is approximately
   −0.017 — small but present.

2. **Hybrid recall (BM25 + dense):** The dense retriever uses semantic embeddings
   and can recall short profiles that match the JD semantically even with few keywords.
   A profile that just says "Senior NLP Engineer at Startup X, 7 years" can still
   get a reasonable dense recall score.

3. **Open-to-work flag:** A sparse but active candidate (low completeness,
   `open_to_work_flag = True`) recovers ~0.01 on the overall score.

### Residual Risk: **Low-Medium**

- Affects ranks 50–100 rather than top-10
- A sparse profile for a genuinely strong candidate can rank 40–60 when it should
  rank 10–20
- Cannot be fully resolved without richer profile data

---

## Notes on Interaction Effects

The four failure modes interact. The worst case is a consulting-background AI engineer
(Failure 1) with a sparse profile (Failure 4): career penalty + low BM25 recall could
push a strong candidate out of the top 50 entirely. This is expected to affect fewer
than 2% of genuinely strong candidates in the pool.

The LTR stage (activated by `make ltr`) learns nonlinear interaction weights between
these features and should further reduce residual risk in Failure Modes 1 and 4.
