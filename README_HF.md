---
title: India Runs - Intelligent Candidate Discovery
emoji: IN
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# Intelligent Candidate Discovery

Built for India Runs Hackathon - Track 1, Data & AI Challenge by Redrob AI.

Paste a job description and get a ranked shortlist from the released candidate
pool with score breakdowns and concise explanations.

Architecture: JD parsing -> BM25 recall -> skill, experience, and behavioral
scoring -> ranked shortlist. The repository also includes LightGBM LTR,
cross-encoder reranking, LLM reranking, and hidden-eval-safe documentation.
