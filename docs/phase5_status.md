# Phase 5 Status

Date: June 2, 2026

## Completed Locally

- Added a Gradio demo package in `app/demo.py` with a Hugging Face entry point
  in `app.py`.
- The demo ranks the released 100,000-candidate pool using BM25 recall plus
  transparent skill, experience, and behavioral scoring.
- Added a Plotly score breakdown for the top candidates.
- Replaced the README with a judge-facing narrative that documents architecture,
  hidden-eval constraints, reproduction commands, and demo usage.
- Added `docs/BLUEPRINT.md`, `docs/DEMO_VIDEO_SCRIPT.md`, and
  `docs/HUGGINGFACE_DEPLOY.md`.
- Prepared `README_HF.md` and `app/requirements.txt` for Hugging Face Spaces.

## Validation

- `python app/demo.py --smoke` completed successfully.
- The smoke run ranked five candidates from 300 recalled profiles using the real
  processed candidate parquet.

## External Items Still Requiring User Action

- Hugging Face Spaces deployment requires user login/token.
- Demo video recording requires the user's Loom, OBS, YouTube, or equivalent
  account.
- Final NDCG/MAP/P@10 reporting requires hidden labels, leaderboard feedback, or
  trusted human relevance judgments.

## Notes for the Next Phase

- Do not fabricate metrics. Keep README results pending until real labels or
  official feedback exist.
- If deploying to Spaces, consider Git LFS or a hosted dataset for the large
  candidate parquet.
- After deployment, paste the live Space URL and video URL into `README.md`.
