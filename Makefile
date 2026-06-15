# India Runs Track 1 — Makefile
# On Windows: use Git Bash or WSL, or run the equivalent Python commands directly.

.PHONY: setup install test reproduce reproduce-full submit demo demo-smoke validate ltr clean help

# ── Default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "India Runs Track 1 — available targets:"
	@echo ""
	@echo "  make setup          Create .venv and install all dependencies"
	@echo "  make test           Run the test suite (pytest)"
	@echo "  make reproduce      Build indices + generate submission end-to-end"
	@echo "  make submit         Generate submission with LLM re-rank enabled"
	@echo "  make validate       Validate an existing submission file"
	@echo "  make demo           Launch the local Gradio demo"
	@echo "  make demo-smoke     Run a quick smoke test of the ranking pipeline"
	@echo "  make clean          Remove cached outputs (models, cache)"
	@echo ""

# ── Environment ───────────────────────────────────────────────────────────────
setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt', 'stopwords', 'wordnet']]"
	@echo ""
	@echo "Setup complete."
	@echo "Activate with:  source .venv/bin/activate    (Linux/macOS)"
	@echo "                .venv\\Scripts\\activate       (Windows)"

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

# ── Full end-to-end reproduction ──────────────────────────────────────────────
reproduce:
	@echo "==> Step 1: Build BM25 retrieval index"
	python scripts/build_indices.py
	@echo ""
	@echo "==> Step 2: Parse job description"
	python scripts/parse_data.py --limit 0
	@echo ""
	@echo "==> Step 3: Generate submission (BM25 + multi-signal scoring)"
	python scripts/generate_submission.py --validate
	@echo ""
	@echo "Reproduction complete."
	@echo "Submission: outputs/submissions/final_submission.csv"
	@echo "To activate LTR for better ranking: make ltr && make reproduce"

# ── Final submission (with optional LLM re-rank) ──────────────────────────────
submit:
	@echo "==> Generating final submission (LLM re-rank enabled)"
	@echo "    Requires a local Ollama server (OLLAMA_BASE_URL / OLLAMA_MODEL in .env)"
	python scripts/generate_submission.py --llm-rerank --validate
	@echo ""
	@echo "Upload outputs/submissions/final_submission.csv to Hack2skill."

# ── LTR training ──────────────────────────────────────────────────────────────
ltr:
	@echo "==> Step 1: Create pseudo-relevance labels"
	python scripts/create_pseudo_labels.py
	@echo "==> Step 2: Generate LTR feature matrix (100K candidates)"
	python scripts/generate_features.py
	@echo "==> Step 3: Train LightGBM LambdaRank"
	python scripts/train_ltr.py
	@echo ""
	@echo "LTR model saved to outputs/models/ltr_model.pkl"
	@echo "Run 'make reproduce' to generate submission with LTR active."

# ── Full pipeline with LTR (~20 min first run) ────────────────────────────────
reproduce-full:
	python scripts/build_indices.py
	python scripts/parse_data.py --limit 0
	python scripts/create_pseudo_labels.py
	python scripts/generate_features.py
	python scripts/train_ltr.py
	python scripts/generate_submission.py --validate
	@echo "Full pipeline complete."

# ── Validate an existing submission ───────────────────────────────────────────
validate:
	python -m src.eval.validate_submission --pred outputs/submissions/final_submission.csv

# ── Demo ──────────────────────────────────────────────────────────────────────
demo:
	python app/demo.py

demo-smoke:
	python app/demo.py --smoke

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	python -c "\
import shutil, pathlib; \
[shutil.rmtree(p, ignore_errors=True) for p in ['outputs/cache', 'outputs/logs']] + \
[p.unlink() for p in pathlib.Path('.').rglob('*.pyc')] + \
[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	@echo "Cache cleared."
