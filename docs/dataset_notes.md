# Dataset Notes

Status as of 2026-06-02: the official public Track 1 bundle is extracted under
`data/raw/india_runs_challenge/`, and the Phase 1 scripts now materialize the
repo-ready canonical datasets in `data/processed/`.

## Job bundle

- Source file: `data/raw/india_runs_challenge/job_description.docx`
- Canonical dataset: `data/processed/challenge_jobs.csv`
- Shape: 1 released job description, 14 flattened columns
- Job ID: `REDROB_TRACK1_MAIN_JD`
- Title: `Senior AI Engineer - Founding Team`
- Company: `Redrob AI (Series A AI-native talent intelligence platform)`
- Location: `Pune/Noida, India (Hybrid...)`
- Experience band: 5.0 to 9.0 years
- Full JD length: 1,514 words
- Biggest JD sections by length:
  - `Let's be honest about this role`: 231 words
  - `What we mean by "5-9 years"`: 207 words
  - `Things we explicitly do NOT want`: 190 words
  - `What you'd actually be doing`: 171 words
  - `Final note for the participants...`: 160 words
- Strongest must-have themes: production retrieval systems, vector/hybrid search
  infrastructure, strong Python, and ranking evaluation rigor.
- Important interpretation note: the JD explicitly warns that keyword matching
  alone is a trap; behavioral availability and real production-system evidence
  matter.

## Candidate pool

- Source file: `data/raw/india_runs_challenge/candidates.jsonl`
- Canonical dataset: `data/processed/challenge_candidates.parquet`
- Shape: 100,000 candidates x 47 flattened columns
- Schema style: nested profile JSON with `profile`, `career_history`,
  `education`, `skills`, and `redrob_signals`, flattened into retrieval- and
  feature-friendly columns.
- Average total experience: 7.166 years
- Median total experience: 6.8 years
- Average skills per candidate: 9.603
- Average profile completeness score: 56.758
- 90th percentile profile completeness score: 80.4
- Median days since last activity: 111
- Average notice period: 87.386 days
- Median notice period: 90 days
- Open-to-work rate: 35.3%
- Verified-email rate: 72.0%
- LinkedIn-connected rate: 36.0%
- Average recruiter response rate: 0.437
- Average skill-assessment average: 51.158
- Work-mode distribution is essentially balanced across `hybrid`, `onsite`,
  `flexible`, and `remote`.
- Top countries:
  - India: 75,113
  - USA: 9,978
  - Australia: 2,579
  - Canada: 2,506
  - UK: 2,472
  - Germany: 2,469
  - Singapore: 2,453
  - UAE: 2,430
- Top current titles are broad and noisy rather than AI-specific, led by
  `Business Analyst`, `HR Manager`, `Mechanical Engineer`, `Accountant`, and
  `Project Manager`.
- Important interpretation note: the sample top-100 file includes many obvious
  non-AI titles, reinforcing the organizer warning that keyword-heavy or
  title-agnostic rankers will get trapped.

## Evaluation and submission contract

- There is no public labels file in the bundle.
- This is a single-query hidden-evaluation challenge, not a public train/val/test benchmark.
- Required submission columns, in order:
  - `candidate_id`
  - `rank`
  - `score`
  - `reasoning`
- Exact row requirement: 100 rows
- Rank requirement: every integer from 1 through 100 exactly once
- Candidate requirement: each `candidate_id` must be unique and present in the
  released pool
- Score requirement: non-increasing as rank gets worse
- Hidden composite metric:
  - `0.50 * NDCG@10`
  - `0.30 * NDCG@50`
  - `0.15 * MAP`
  - `0.05 * P@10`
- Honeypot warning: the organizers explicitly state that a small set of subtly
  impossible profiles are forced to relevance tier 0 and can disqualify brittle
  keyword-only systems.

## Phase 1 implications

- Public hidden-eval means we cannot create meaningful local train/val/test
  splits from the released bundle.
- Phase 1 is therefore redefined in this repo as:
  - extract and normalize the public bundle
  - run real-data EDA on the JD and candidates
  - confirm the exact submission contract
  - validate sample and local submission files before upload
- Local scoring remains blocked until official labels are ever released.
