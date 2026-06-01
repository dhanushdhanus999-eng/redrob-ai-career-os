# India Runs · Track 1

Dataset-ready foundation for the Redrob AI "Intelligent Candidate Discovery"
challenge.

## Current status

Phase 1 is implemented as far as the currently available materials allow:

- repository scaffold and package layout
- pinned project configuration in `pyproject.toml`
- evaluation harness with ranking metrics and tests
- dataset discovery, loading, and split utilities
- EDA scripts for jobs, candidates, and labels
- documentation, memory notes, and a Phase 1 session log

The official dataset has not been released into this workspace yet, so the
dataset-dependent notebooks and split generation are ready to run but have not
been executed.

## Quick start

1. Create a Python 3.11 virtual environment.
2. Install the project in editable mode:

   ```bash
   pip install -e ".[dev]"
   ```

3. Copy the official dataset into `data/raw/`.
4. Run the dataset exploration scripts in `notebooks/`.
5. Create splits and baseline artifacts:

   ```bash
   python notebooks/04_create_splits_and_baseline.py
   ```

6. Score any ranked output file:

   ```bash
   python -m src.eval.score_submission --pred outputs/submissions/your_file.csv
   ```

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
