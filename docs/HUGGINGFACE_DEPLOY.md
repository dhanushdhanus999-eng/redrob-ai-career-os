# Hugging Face Spaces Deployment Notes

The repository is ready for a Gradio Space. The actual deployment still needs a
Hugging Face account/token from the user.

## Files Prepared

- `app.py` - Spaces entry point
- `app/demo.py` - Gradio Blocks UI and ranking logic
- `app/requirements.txt` - lightweight demo dependencies
- `README_HF.md` - Space card metadata and description

## Deploy Steps

```powershell
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create india-runs-ranking --type space --space_sdk gradio
```

Copy or push these files to the Space repo:

```text
app.py
app/
src/
data/processed/challenge_jobs.csv
data/processed/challenge_candidates.parquet
README_HF.md as README.md
```

The first app run builds `outputs/models/bm25_demo_index.pkl`. That cache is not
checked into git, but the Space can regenerate it from the processed candidate
parquet.

## Post-Deploy Checks

1. Open the Space URL and wait for the first build to finish.
2. Run the default Senior AI Engineer JD.
3. Confirm the table returns candidates and the Plotly score breakdown renders.
4. Paste a different JD and confirm the ranking changes.
5. Add the final Space URL to the main `README.md`.

## Known Blockers

- Deployment requires user-controlled Hugging Face credentials.
- The processed candidate parquet is large, so the Space may need Git LFS or a
  dataset-hosting strategy if regular git push rejects the file size.
- Hidden-eval metrics cannot be displayed until labels or leaderboard feedback
  are available.
