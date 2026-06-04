# Phase 1 Status

## Completed

- Project scaffold and directory structure created
- `pyproject.toml`, `.env.example`, `.gitignore`, and MIT license added
- README skeleton created
- Ranking metrics, split logic, submission scoring CLI, and tests added
- Public-bundle discovery, DOCX extraction, and candidate-flattening utilities added
- Released dataset extracted to `data/raw/india_runs_challenge/`
- Canonical processed datasets created:
  - `data/processed/challenge_jobs.csv`
  - `data/processed/challenge_candidates.parquet`
- Phase 1 scripts updated to the real single-JD hidden-eval challenge format
- Sample-submission validation CLI added: `python -m src.eval.validate_submission`
- Memory notes and session documentation added

## No longer blocked in Phase 1

- Dataset download and inspection
- Actual EDA outputs and figures
- Submission-format confirmation from released organizer files
- Bundle validation against the released sample submission

## Open questions for later phases

- No public labels or leaderboard exist, so local ranking quality remains unscored
- The best Phase 2 and Phase 3 model variants still need to be chosen by
  offline reasoning plus final hidden-eval results
- Honeypot avoidance and behavioral-signal weighting need to be pressure-tested
  in real ranking runs
