"""India Runs Track 1 interactive ranking demo.

Paste a job description and get ranked candidates with transparent score
breakdowns. The demo uses the released public bundle and avoids pretending that
hidden evaluation labels or trained LTR artifacts exist locally.

Run a core smoke test:
    python app/demo.py --smoke

Run the Gradio UI:
    python app/demo.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import build_candidate_documents, load_phase2_bundle
from src.data.schema import combine_text_values
from src.parsing.candidate_parser import CandidateProfileParser
from src.parsing.jd_parser import JobDescriptionParser
from src.ranking.explainer import explain_ranking
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.paths import MODELS_DIR
from src.utils.role_relevance import score_career_trajectory, score_role_relevance
from src.utils.skill_ontology import SKILL_FAMILIES, SKILL_SYNONYMS, SkillMatcher, normalize_skill


BM25_DEMO_INDEX = MODELS_DIR / "bm25_demo_index.pkl"
DEFAULT_RECALL_K = 300
SCORE_COLUMNS = ["Retrieval", "Skill", "Experience Fit", "Behavioral", "Role Fit"]


EXAMPLE_JD = """Senior AI Engineer - Founding Team

We are hiring a senior engineer to own the intelligence layer of a talent
platform. The role needs production experience with Python, embeddings-based
retrieval, vector databases, hybrid search, ranking evaluation, and LLM-powered
reranking.

Requirements:
- 5-9 years of applied ML or search/recommendation experience
- Strong Python and production engineering judgment
- Hands-on retrieval systems using BGE, E5, OpenAI embeddings, FAISS, Qdrant,
  Pinecone, Weaviate, Milvus, OpenSearch, or Elasticsearch
- Experience evaluating ranking systems with NDCG, MAP, MRR, A/B tests, and
  recruiter feedback loops
- Bonus: LoRA, PEFT, learning-to-rank, marketplace products, or HR-tech
"""


DEMO_TERMS = {
    "A/B Testing",
    "Artificial Intelligence",
    "BGE",
    "CI/CD",
    "Docker",
    "E5",
    "Elasticsearch",
    "Embeddings",
    "FAISS",
    "FastAPI",
    "Feature Engineering",
    "Google Cloud Platform",
    "Hybrid Search",
    "Kubernetes",
    "Large Language Models",
    "Learning to Rank",
    "LightGBM",
    "LoRA",
    "Machine Learning",
    "MAP",
    "Milvus",
    "MRR",
    "Natural Language Processing",
    "NDCG",
    "OpenAI Embeddings",
    "OpenSearch",
    "PEFT",
    "Pinecone",
    "Python",
    "Qdrant",
    "QLoRA",
    "Ranking",
    "Recommendation Systems",
    "Retrieval",
    "Sentence Transformers",
    "Vector Databases",
    "Weaviate",
}


# ─────────────────────────────────────────────────────────────────────────────
# UI ASSETS — CSS / JS / HTML
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=Space+Grotesk:wght@300;400;500;600;700;900&display=swap');

/* ── Variables ─────────────────────────────────────────────── */
:root {
    --ir-bg:        #020817;
    --ir-card:      rgba(10, 18, 38, 0.82);
    --ir-border:    rgba(59, 130, 246, 0.18);
    --ir-blue:      #3b82f6;
    --ir-blue-l:    #60a5fa;
    --ir-cyan:      #22d3ee;
    --ir-purple:    #a855f7;
    --ir-green:     #10b981;
    --ir-gold:      #f59e0b;
    --ir-text:      #f1f5f9;
    --ir-muted:     #64748b;
    --ir-dim:       #94a3b8;
}

/* ── Reset / Body ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body {
    background: var(--ir-bg) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--ir-text) !important;
    margin: 0 !important;
}

/* ── Gradio shell ───────────────────────────────────────────── */
.gradio-container {
    background: transparent !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
.main, .wrap, footer { background: transparent !important; }
.contain { padding: 0 !important; }

/* Hide default Gradio footer */
footer { display: none !important; }

/* ── Hero ───────────────────────────────────────────────────── */
.ir-hero {
    position: relative;
    padding: 64px 40px 48px;
    text-align: center;
    overflow: hidden;
    border-bottom: 1px solid rgba(59,130,246,0.12);
}

.ir-hero-bg {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
        radial-gradient(ellipse 60% 50% at 20% 50%, rgba(59,130,246,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 50% 60% at 80% 30%, rgba(168,85,247,0.10) 0%, transparent 70%),
        radial-gradient(ellipse 40% 40% at 55% 85%, rgba(34,211,238,0.08) 0%, transparent 70%);
    animation: heroBgPulse 8s ease-in-out infinite alternate;
}

@keyframes heroBgPulse {
    0%   { opacity: 0.6; }
    100% { opacity: 1.0; }
}

.ir-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 18px;
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.28);
    border-radius: 100px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: var(--ir-blue-l);
    text-transform: uppercase;
    margin-bottom: 24px;
}

.ir-badge-dot {
    width: 7px; height: 7px;
    background: var(--ir-green);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--ir-green);
    animation: dotBlink 2s ease-in-out infinite;
}

@keyframes dotBlink {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--ir-green); }
    50%       { opacity: 0.4; box-shadow: none; }
}

.ir-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(52px, 9vw, 108px) !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    margin: 0 0 18px !important;
    background: linear-gradient(130deg,
        #ffffff   0%,
        #93c5fd  25%,
        #22d3ee  55%,
        #c084fc  80%,
        #f0abfc 100%);
    background-size: 250% auto;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: titleShimmer 5s linear infinite;
}

@keyframes titleShimmer {
    0%   { background-position: 0%   center; }
    100% { background-position: 250% center; }
}

.ir-subtitle {
    font-size: clamp(14px, 2vw, 18px) !important;
    color: var(--ir-dim) !important;
    font-weight: 300 !important;
    letter-spacing: 0.06em !important;
    margin: 0 0 44px !important;
    min-height: 28px;
}

.ir-stats-row {
    display: flex;
    justify-content: center;
    align-items: stretch;
    gap: 16px;
    flex-wrap: wrap;
}

.ir-stat-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 18px 32px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    cursor: default;
    transition: border-color 0.35s, background 0.35s, transform 0.35s, box-shadow 0.35s;
}
.ir-stat-card:hover {
    border-color: rgba(59,130,246,0.45);
    background: rgba(59,130,246,0.07);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(59,130,246,0.18);
}

.ir-stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
}
.ir-stat-label {
    font-size: 11px;
    color: var(--ir-muted);
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
.ir-stat-card:nth-child(1) .ir-stat-value { color: var(--ir-blue-l); }
.ir-stat-card:nth-child(2) .ir-stat-value { color: var(--ir-cyan); }
.ir-stat-card:nth-child(3) .ir-stat-value { color: var(--ir-green); }
.ir-stat-card:nth-child(4) .ir-stat-value { color: var(--ir-gold); }

/* ── Section labels ─────────────────────────────────────────── */
.ir-section-label {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ir-blue-l);
    margin-bottom: 10px;
    margin-top: 4px;
}
.ir-section-label::before {
    content: '';
    display: block;
    width: 24px; height: 2px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--ir-blue), transparent);
}

/* ── Main layout padding ────────────────────────────────────── */
.gradio-container > .main > .wrap > .gap {
    padding: 32px 28px !important;
    gap: 24px !important;
}

/* ── Blocks / cards ─────────────────────────────────────────── */
.block {
    background: var(--ir-card) !important;
    border: 1px solid var(--ir-border) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
.block:hover {
    border-color: rgba(59,130,246,0.32) !important;
    box-shadow: 0 8px 48px rgba(59,130,246,0.10) !important;
}

/* ── Labels ─────────────────────────────────────────────────── */
label > span,
.label-wrap span,
.svelte-1ipelgc {
    color: var(--ir-dim) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Textarea ───────────────────────────────────────────────── */
textarea {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: var(--ir-text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    line-height: 1.75 !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
textarea:focus {
    border-color: rgba(59,130,246,0.55) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12), inset 0 0 24px rgba(59,130,246,0.03) !important;
    outline: none !important;
}
textarea::-webkit-scrollbar { width: 5px; }
textarea::-webkit-scrollbar-track { background: transparent; }
textarea::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.28); border-radius: 3px; }

/* ── Slider ─────────────────────────────────────────────────── */
input[type="range"] { accent-color: var(--ir-blue) !important; }

input[type="number"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 8px !important;
    color: var(--ir-text) !important;
}

/* ── Primary button ─────────────────────────────────────────── */
button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 55%, #06b6d4 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.25s, box-shadow 0.25s !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.45) !important;
    animation: btnGlow 3.5s ease-in-out infinite !important;
}
button.primary::after {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(120deg, transparent 30%, rgba(255,255,255,0.18) 50%, transparent 70%) !important;
    transform: translateX(-120%) !important;
    transition: transform 0.6s ease !important;
}
button.primary:hover::after { transform: translateX(120%) !important; }
button.primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 40px rgba(59,130,246,0.65) !important;
}
button.primary:active { transform: translateY(0) scale(0.98) !important; }

@keyframes btnGlow {
    0%, 100% { box-shadow: 0 4px 24px rgba(59,130,246,0.45); }
    50%       { box-shadow: 0 4px 36px rgba(59,130,246,0.80), 0 0 64px rgba(59,130,246,0.20); }
}

/* ── Status textbox ─────────────────────────────────────────── */
#status-box textarea {
    background: rgba(16,185,129,0.04) !important;
    border-color: rgba(16,185,129,0.18) !important;
    color: #6ee7b7 !important;
    font-family: 'Space Grotesk', monospace !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
}

/* ── Dataframe ──────────────────────────────────────────────── */
.table-wrap {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(59,130,246,0.14) !important;
}
table { background: rgba(5,12,28,0.55) !important; }

thead tr { background: rgba(59,130,246,0.09) !important; }
thead th {
    color: var(--ir-blue-l) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(59,130,246,0.18) !important;
    padding: 12px 10px !important;
    white-space: nowrap !important;
}

tbody tr {
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
    transition: background 0.18s !important;
}
tbody tr:hover { background: rgba(59,130,246,0.07) !important; }

tbody td {
    color: var(--ir-text) !important;
    font-size: 12px !important;
    padding: 10px 10px !important;
    border: none !important;
}
tbody td:first-child {
    color: var(--ir-gold) !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
}

/* scrollbar inside dataframe */
.table-wrap::-webkit-scrollbar { width: 6px; height: 6px; }
.table-wrap::-webkit-scrollbar-track { background: transparent; }
.table-wrap::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.25); border-radius: 4px; }

/* ── Plotly chart ───────────────────────────────────────────── */
.js-plotly-plot { border-radius: 12px !important; }

/* ── Progress / loading ─────────────────────────────────────── */
.generating, .eta-bar {
    background: rgba(59,130,246,0.08) !important;
    border-color: rgba(59,130,246,0.25) !important;
    border-radius: 8px !important;
}

/* ── Footer ─────────────────────────────────────────────────── */
.ir-footer {
    text-align: center;
    padding: 28px 24px 40px;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 8px;
}
.ir-pipeline-strip {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
    margin-bottom: 16px;
}
.ir-pipe-step {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 18px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    font-size: 11px;
    color: var(--ir-dim);
    letter-spacing: 0.06em;
    transition: background 0.2s, color 0.2s;
}
.ir-pipe-step:first-child { border-radius: 100px 0 0 100px; }
.ir-pipe-step:last-child  { border-radius: 0 100px 100px 0; }
.ir-pipe-step:hover { background: rgba(59,130,246,0.08); color: var(--ir-blue-l); }

.ir-pipe-arrow {
    color: rgba(59,130,246,0.35);
    font-size: 16px;
    line-height: 1;
    margin: 0 -1px;
}
.ir-pipe-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}

.ir-footer-copy {
    font-size: 12px;
    color: var(--ir-muted);
    opacity: 0.55;
    margin: 0;
    letter-spacing: 0.05em;
}

/* ── Entry animation ────────────────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.gradio-container > .main > .wrap > .gap > * {
    animation: fadeUp 0.55s ease both;
}
.gradio-container > .main > .wrap > .gap > *:nth-child(1) { animation-delay: 0.05s; }
.gradio-container > .main > .wrap > .gap > *:nth-child(2) { animation-delay: 0.15s; }
.gradio-container > .main > .wrap > .gap > *:nth-child(3) { animation-delay: 0.25s; }
"""


CUSTOM_JS = """
() => {
    // Give Svelte a moment to mount before we inject
    setTimeout(function () {

        // ── Particle canvas ────────────────────────────────────
        var cv = document.createElement('canvas');
        cv.style.cssText = [
            'position:fixed', 'top:0', 'left:0',
            'width:100%', 'height:100%',
            'z-index:0', 'pointer-events:none', 'opacity:0.55'
        ].join(';');
        document.body.insertBefore(cv, document.body.firstChild);

        var ctx = cv.getContext('2d');
        var W, H;

        function resize() {
            W = cv.width  = window.innerWidth;
            H = cv.height = window.innerHeight;
        }
        resize();
        window.addEventListener('resize', resize);

        // 70 nodes
        var N = 70;
        var pts = [];
        for (var i = 0; i < N; i++) {
            pts.push({
                x: Math.random() * W,
                y: Math.random() * H,
                vx: (Math.random() - 0.5) * 0.45,
                vy: (Math.random() - 0.5) * 0.45,
                r: Math.random() * 1.8 + 0.5,
                hue: Math.random() > 0.55 ? 214 : 186   // blue / cyan
            });
        }

        var LINK = 130;

        function frame() {
            ctx.clearRect(0, 0, W, H);

            // connections
            for (var i = 0; i < N; i++) {
                for (var j = i + 1; j < N; j++) {
                    var dx = pts[i].x - pts[j].x;
                    var dy = pts[i].y - pts[j].y;
                    var d2 = dx * dx + dy * dy;
                    if (d2 < LINK * LINK) {
                        var alpha = 0.14 * (1 - Math.sqrt(d2) / LINK);
                        ctx.strokeStyle = 'rgba(59,130,246,' + alpha + ')';
                        ctx.lineWidth = 0.8;
                        ctx.beginPath();
                        ctx.moveTo(pts[i].x, pts[i].y);
                        ctx.lineTo(pts[j].x, pts[j].y);
                        ctx.stroke();
                    }
                }
            }

            // dots + glow
            for (var i = 0; i < N; i++) {
                var p = pts[i];
                var g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 5);
                g.addColorStop(0, 'hsla(' + p.hue + ',88%,65%,0.55)');
                g.addColorStop(1, 'hsla(' + p.hue + ',88%,65%,0)');
                ctx.fillStyle = g;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r * 5, 0, 6.2832);
                ctx.fill();

                ctx.fillStyle = 'hsla(' + p.hue + ',90%,72%,0.88)';
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, 6.2832);
                ctx.fill();

                p.x += p.vx; p.y += p.vy;
                if (p.x < 0 || p.x > W) p.vx *= -1;
                if (p.y < 0 || p.y > H) p.vy *= -1;
            }

            requestAnimationFrame(frame);
        }
        frame();

        // ── Typing effect on subtitle ──────────────────────────
        var sub = document.querySelector('.ir-subtitle');
        if (sub && sub.textContent.trim()) {
            var full = sub.textContent.trim();
            sub.textContent = '';
            sub.style.opacity = '1';
            var idx = 0;
            function type() {
                if (idx < full.length) {
                    sub.textContent += full[idx++];
                    setTimeout(type, 32);
                }
            }
            setTimeout(type, 500);
        }

        // ── Stagger-fade stat cards ────────────────────────────
        var cards = document.querySelectorAll('.ir-stat-card');
        cards.forEach(function (el, i) {
            el.style.opacity = '0';
            el.style.transform = 'translateY(14px)';
            el.style.transition = 'opacity 0.5s ease ' + (0.3 + i * 0.12) + 's, transform 0.5s ease ' + (0.3 + i * 0.12) + 's';
            setTimeout(function () {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, 50);
        });

    }, 350);
}
"""


HERO_HTML = """
<div class="ir-hero">
    <div class="ir-hero-bg"></div>
    <div style="position:relative;z-index:2">
        <div class="ir-badge">
            <span class="ir-badge-dot"></span>
            Track 1 &nbsp;·&nbsp; Live Ranking Demo
        </div>
        <h1 class="ir-title">India Runs</h1>
        <p class="ir-subtitle">Intelligent Candidate Discovery Platform</p>
        <div class="ir-stats-row">
            <div class="ir-stat-card">
                <span class="ir-stat-value">100K+</span>
                <span class="ir-stat-label">Candidates</span>
            </div>
            <div class="ir-stat-card">
                <span class="ir-stat-value">8</span>
                <span class="ir-stat-label">Scoring Signals</span>
            </div>
            <div class="ir-stat-card">
                <span class="ir-stat-value">&lt;100ms</span>
                <span class="ir-stat-label">Ranking Speed</span>
            </div>
            <div class="ir-stat-card">
                <span class="ir-stat-value">BM25+</span>
                <span class="ir-stat-label">Retrieval Engine</span>
            </div>
        </div>
    </div>
</div>
"""


FOOTER_HTML = """
<div class="ir-footer">
    <div class="ir-pipeline-strip">
        <div class="ir-pipe-step">
            <span class="ir-pipe-dot" style="background:#3b82f6"></span>JD Parsing
        </div>
        <span class="ir-pipe-arrow">›</span>
        <div class="ir-pipe-step">
            <span class="ir-pipe-dot" style="background:#22d3ee"></span>BM25 Recall
        </div>
        <span class="ir-pipe-arrow">›</span>
        <div class="ir-pipe-step">
            <span class="ir-pipe-dot" style="background:#a855f7"></span>Skill Match
        </div>
        <span class="ir-pipe-arrow">›</span>
        <div class="ir-pipe-step">
            <span class="ir-pipe-dot" style="background:#10b981"></span>Multi-Signal Scoring
        </div>
        <span class="ir-pipe-arrow">›</span>
        <div class="ir-pipe-step">
            <span class="ir-pipe-dot" style="background:#f59e0b"></span>Explainable Ranking
        </div>
    </div>
    <p class="ir-footer-copy">India Runs &nbsp;·&nbsp; Track 1 &nbsp;·&nbsp; Intelligent Candidate Discovery</p>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CandidateScore:
    """Score details for one retrieved candidate."""

    candidate_id: str
    retrieval_score: float
    skill_score: float
    experience_score: float
    behavioral_score: float
    role_score: float
    career_score: float
    location_score: float
    overall_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    rationale: str


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_skills(value: object) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    skills: list[str] = []
    seen: set[str] = set()
    for raw_part in text.replace(";", ",").replace("|", ",").split(","):
        part = raw_part.strip(" .:-")
        if not part:
            continue
        canonical = normalize_skill(part)
        lowered = canonical.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        skills.append(canonical)
    return skills


def _term_variants() -> list[tuple[str, str]]:
    terms = set(DEMO_TERMS)
    terms.update(SKILL_SYNONYMS)
    terms.update(SKILL_SYNONYMS.values())
    for family_skills in SKILL_FAMILIES.values():
        terms.update(family_skills)

    variants: list[tuple[str, str]] = []
    for term in sorted(terms, key=len, reverse=True):
        canonical = normalize_skill(term)
        variants.append((term.lower(), canonical))
    return variants


TERM_VARIANTS = _term_variants()


def extract_skill_mentions(text: str, *, max_skills: int = 16) -> list[str]:
    """Extract known technical and ranking terms from a job description."""
    lowered = f" {_safe_text(text).lower()} "
    found: list[str] = []
    seen: set[str] = set()

    for variant, canonical in TERM_VARIANTS:
        if not variant or canonical.lower() in seen:
            continue
        if variant in lowered:
            seen.add(canonical.lower())
            found.append(canonical)
        if len(found) >= max_skills:
            break

    return found


def score_location(candidate_location: str) -> float:
    """Score candidate location fit for a Pune/Noida hybrid role in India."""
    loc = str(candidate_location).lower().strip()
    india_tokens = (
        "india", "pune", "noida", "bangalore", "bengaluru",
        "hyderabad", "mumbai", "delhi", "chennai", "gurugram",
        "gurgaon", "kolkata", "ahmedabad", "jaipur", "kochi",
    )
    if any(t in loc for t in india_tokens):
        return 1.0
    if not loc or loc in ("nan", "none", "not specified", ""):
        return 0.55
    return 0.25


def score_experience(candidate_years: float, min_years: float | None, max_years: float | None) -> float:
    """Score candidate experience against a parsed JD range."""
    if not min_years and not max_years:
        return 0.5
    if min_years and candidate_years < min_years:
        return round(max(0.0, candidate_years / max(min_years, 1.0)), 4)
    if max_years and candidate_years > max_years:
        extra_years = candidate_years - max_years
        return round(max(0.45, 1.0 - extra_years / max(max_years * 2.0, 1.0)), 4)
    return 1.0


def score_behavior(row: pd.Series) -> float:
    """Combine public behavioral/activity signals into a 0-1 score."""
    completeness = _safe_float(row.get("profile_completeness_score")) / 100.0
    if completeness == 0.0:
        completeness = _safe_float(row.get("profile_completeness"))

    response_rate = _safe_float(row.get("recruiter_response_rate"))
    github_activity = min(_safe_float(row.get("github_activity_score")) / 100.0, 1.0)
    saved_by_recruiters = min(_safe_float(row.get("saved_by_recruiters_30d")) / 10.0, 1.0)
    search_appearance = min(_safe_float(row.get("search_appearance_30d")) / 500.0, 1.0)
    assessment_avg = min(_safe_float(row.get("skill_assessment_avg")) / 100.0, 1.0)
    open_to_work = 1.0 if bool(row.get("open_to_work_flag")) else 0.0

    recency = 0.4
    raw_last_active = _safe_text(row.get("last_active"))
    if raw_last_active:
        try:
            active_date = datetime.fromisoformat(raw_last_active[:10]).date()
            days_since_active = max((date.today() - active_date).days, 0)
            recency = max(0.0, 1.0 - days_since_active / 180.0)
        except ValueError:
            recency = 0.4

    score = (
        0.25 * completeness
        + 0.20 * recency
        + 0.20 * response_rate
        + 0.10 * open_to_work
        + 0.10 * github_activity
        + 0.05 * saved_by_recruiters
        + 0.05 * search_appearance
        + 0.05 * assessment_avg
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def score_breakdown_chart(result_df: pd.DataFrame | None) -> go.Figure:
    """Build a grouped bar chart for the top candidates — dark theme."""
    base_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(l=32, r=24, t=60, b=68),
        height=340,
    )

    figure = go.Figure()
    if result_df is None or result_df.empty:
        figure.update_layout(title="Score Breakdown", **base_layout)
        return figure

    top_rows = result_df.head(5)
    palette = ["#3b82f6", "#22d3ee", "#a855f7", "#10b981", "#f59e0b"]

    for i, column in enumerate(SCORE_COLUMNS):
        figure.add_trace(
            go.Bar(
                name=column,
                x=top_rows["Candidate ID"],
                y=top_rows[column],
                text=[f"{v:.0%}" for v in top_rows[column]],
                textposition="outside",
                textfont=dict(color="#cbd5e1", size=10, family="Space Grotesk, sans-serif"),
                marker=dict(
                    color=palette[i],
                    opacity=0.88,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    f"<b>{column}</b><br>"
                    "Candidate: %{x}<br>"
                    "Score: %{y:.3f}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=dict(
            text="Score Breakdown — Top 5 Candidates",
            font=dict(color="#f1f5f9", size=13, family="Space Grotesk, sans-serif"),
        ),
        barmode="group",
        yaxis=dict(
            range=[0, 1.25],
            tickformat=".0%",
            gridcolor="rgba(59,130,246,0.07)",
            color="#475569",
            tickfont=dict(size=10),
            zeroline=False,
        ),
        xaxis=dict(
            color="#475569",
            tickfont=dict(size=10),
            tickangle=-15,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#94a3b8", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="rgba(10,18,38,0.92)",
            bordercolor="rgba(59,130,246,0.35)",
            font=dict(family="Inter", color="white"),
        ),
        **base_layout,
    )
    return figure


class DemoRankingEngine:
    """Runtime state for the local/Hugging Face demo."""

    def __init__(self) -> None:
        self.bundle = load_phase2_bundle(require_labels=False)
        self.jobs = self.bundle.jobs
        self.candidates = self.bundle.candidates.copy()
        candidate_id_col = self.bundle.candidate_schema.candidate_id
        self.candidates["_candidate_id_str"] = self.candidates[candidate_id_col].astype(str)
        self.candidate_lookup = self.candidates.set_index("_candidate_id_str", drop=False)
        self.job_parser = JobDescriptionParser()
        self.candidate_parser = CandidateProfileParser()
        self.skill_matcher = SkillMatcher()
        self.bm25 = self._load_or_build_bm25()

    def _load_or_build_bm25(self) -> BM25Retriever:
        retriever = BM25Retriever()
        if BM25_DEMO_INDEX.exists():
            try:
                retriever.load(BM25_DEMO_INDEX)
                return retriever
            except Exception:
                BM25_DEMO_INDEX.unlink(missing_ok=True)

        candidate_ids, documents = build_candidate_documents(self.bundle)
        retriever.build_index(documents=documents, candidate_ids=candidate_ids)
        retriever.save(BM25_DEMO_INDEX)
        return retriever

    def default_job_description(self) -> str:
        if self.jobs.empty:
            return EXAMPLE_JD
        first_job = self.jobs.iloc[0]
        job_text = combine_text_values(first_job, self.bundle.job_schema.text_columns)
        return job_text or EXAMPLE_JD

    def parse_job(self, job_description: str) -> dict[str, Any]:
        parsed = self.job_parser.parse_text(job_description)
        extracted_skills = extract_skill_mentions(job_description)
        if extracted_skills:
            parsed["must_have_skills"] = extracted_skills
        return parsed

    def _candidate_row(self, candidate_id: str) -> pd.Series:
        selected = self.candidate_lookup.loc[str(candidate_id)]
        if isinstance(selected, pd.DataFrame):
            return selected.iloc[0]
        return selected

    def _score_candidate(
        self,
        *,
        rank_position: int,
        parsed_job: dict[str, Any],
        candidate_id: str,
        retrieval_norm: float,
    ) -> CandidateScore:
        row = self._candidate_row(candidate_id)
        parsed_candidate = self.candidate_parser.parse_row(row, self.bundle.candidate_schema)

        candidate_skills = parsed_candidate.get("skills") or _split_skills(row.get("skills"))
        must_skills = list(parsed_job.get("must_have_skills", []))
        nice_skills = list(parsed_job.get("nice_to_have_skills", []))

        must_match  = self.skill_matcher.match_score(must_skills, candidate_skills)
        nice_match  = self.skill_matcher.match_score(nice_skills, candidate_skills)
        skill_score = must_match["composite_score"] * 0.75 + nice_match["composite_score"] * 0.25

        candidate_years = _safe_float(
            parsed_candidate.get("total_experience_years"),
            default=_safe_float(row.get("total_experience")),
        )
        experience    = score_experience(
            candidate_years,
            parsed_job.get("min_years_experience"),
            parsed_job.get("max_years_experience"),
        )
        behavioral    = score_behavior(row)
        role          = score_role_relevance(
            _safe_text(row.get("current_role")),
            _safe_text(row.get("headline")),
        )
        career        = score_career_trajectory(_safe_text(row.get("career_history_text")))
        cand_location = _safe_text(row.get("location")) or _safe_text(row.get("country"))
        location      = score_location(cand_location)

        # No semantic model in demo — use no-semantic weight redistribution
        # (matches generate_submission.py --no-semantic formula)
        overall = (
            0.23 * retrieval_norm
            + 0.27 * skill_score
            + 0.15 * role
            + 0.12 * experience
            + 0.11 * behavioral
            + 0.07 * career
            + 0.05 * location
        )
        rationale = explain_ranking(
            rank=rank_position,
            job_title=str(parsed_job.get("title", "")),
            must_skills=must_skills,
            nice_skills=nice_skills,
            candidate_skills=list(candidate_skills),
            cand_years_exp=candidate_years,
            job_min_years=_safe_float(parsed_job.get("min_years_experience")),
            seniority_match=(
                str(parsed_job.get("seniority", "")).lower()
                == str(parsed_candidate.get("seniority", "")).lower()
            ),
            behavioral_score=behavioral,
            semantic_score=0.0,  # no semantic model in demo
        )

        return CandidateScore(
            candidate_id=str(candidate_id),
            retrieval_score=round(retrieval_norm, 4),
            skill_score=round(skill_score, 4),
            experience_score=round(experience, 4),
            behavioral_score=round(behavioral, 4),
            role_score=round(role, 4),
            career_score=round(career, 4),
            location_score=round(location, 4),
            overall_score=round(float(overall), 4),
            matched_skills=list(must_match.get("matched_skills", [])),
            missing_skills=list(must_match.get("missing_skills", [])),
            rationale=rationale,
        )

    def rank(self, job_description: str, top_k: int = 10) -> tuple[pd.DataFrame, str, go.Figure]:
        job_description = _safe_text(job_description)
        if not job_description:
            empty = pd.DataFrame()
            return empty, "Paste a job description to start ranking.", score_breakdown_chart(empty)

        start_time = time.perf_counter()
        parsed_job = self.parse_job(job_description)
        recall_k = max(DEFAULT_RECALL_K, min(1000, int(top_k) * 40))
        recall_results = self.bm25.retrieve(job_description, top_k=recall_k)
        if not recall_results:
            empty = pd.DataFrame()
            return empty, "No candidates retrieved for this job description.", score_breakdown_chart(empty)

        raw_scores = np.asarray([score for _, score in recall_results], dtype=float)
        score_max = float(raw_scores.max()) or 1e-9

        scored_candidates: list[CandidateScore] = []
        for index, (candidate_id, raw_score) in enumerate(recall_results, start=1):
            retrieval_norm = float(raw_score) / score_max
            scored_candidates.append(
                self._score_candidate(
                    rank_position=index,
                    parsed_job=parsed_job,
                    candidate_id=candidate_id,
                    retrieval_norm=retrieval_norm,
                )
            )

        ranked = sorted(scored_candidates, key=lambda item: item.overall_score, reverse=True)[: int(top_k)]
        output_rows: list[dict[str, object]] = []
        for rank, score in enumerate(ranked, start=1):
            row = self._candidate_row(score.candidate_id)
            skills = _split_skills(row.get("skills"))
            output_rows.append(
                {
                    "Rank": rank,
                    "Candidate ID": score.candidate_id,
                    "Current Role": _safe_text(row.get("current_role")),
                    "Company": _safe_text(row.get("current_company")),
                    "Location": _safe_text(row.get("location")) or _safe_text(row.get("country")),
                    "Experience (yrs)": round(_safe_float(row.get("total_experience")), 1),
                    "Overall": score.overall_score,
                    "Retrieval": score.retrieval_score,
                    "Skill": score.skill_score,
                    "Experience Fit": score.experience_score,
                    "Behavioral": score.behavioral_score,
                    "Role Fit": score.role_score,
                    "Matched Skills": ", ".join(score.matched_skills[:5]) or "-",
                    "Missing Skills": ", ".join(score.missing_skills[:5]) or "-",
                    "Top Profile Skills": ", ".join(skills[:6]) or "-",
                    "Why This Rank": score.rationale,
                }
            )

        result_df = pd.DataFrame(output_rows)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        skills_preview = ", ".join(parsed_job.get("must_have_skills", [])[:6]) or "none inferred"
        status = (
            f"Ranked {len(result_df)} candidates from {len(recall_results)} recalled profiles "
            f"in {elapsed_ms:.0f} ms  ·  Parsed skills: {skills_preview}  ·  "
            "Scoring: BM25 + skill + experience + behavioral signals"
        )
        return result_df, status, score_breakdown_chart(result_df)


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE: DemoRankingEngine | None = None


def get_engine() -> DemoRankingEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DemoRankingEngine()
    return _ENGINE


def rank_candidates(job_description: str, top_k: int = 10) -> tuple[pd.DataFrame, str, go.Figure]:
    """Gradio-compatible ranking function."""
    return get_engine().rank(job_description, int(top_k))


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO APP
# ─────────────────────────────────────────────────────────────────────────────

def build_demo() -> Any:
    """Create the Gradio Blocks app."""
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Gradio is not installed. Run: pip install -r app/requirements.txt"
        ) from exc

    default_jd = get_engine().default_job_description()

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Space Grotesk"), "monospace"],
    ).set(
        body_background_fill="#020817",
        body_background_fill_dark="#020817",
        block_background_fill="rgba(10,18,38,0.82)",
        block_background_fill_dark="rgba(10,18,38,0.82)",
        block_border_width="1px",
        block_border_color="rgba(59,130,246,0.18)",
        block_border_color_dark="rgba(59,130,246,0.18)",
        input_background_fill="rgba(255,255,255,0.03)",
        input_background_fill_dark="rgba(255,255,255,0.03)",
        input_border_color="rgba(255,255,255,0.09)",
        button_primary_background_fill="linear-gradient(135deg,#1d4ed8 0%,#3b82f6 55%,#06b6d4 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg,#2563eb 0%,#60a5fa 55%,#22d3ee 100%)",
        button_primary_text_color="#ffffff",
        body_text_color="#f1f5f9",
        body_text_color_subdued="#94a3b8",
        block_label_text_color="#64748b",
    )

    with gr.Blocks(
        title="India Runs — Intelligent Candidate Discovery",
        theme=theme,
        css=CUSTOM_CSS,
        js=CUSTOM_JS,
    ) as demo:

        # ── Hero ────────────────────────────────────────────────
        gr.HTML(HERO_HTML)

        # ── Main ────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── Left — Input ────────────────────────────────────
            with gr.Column(scale=1, min_width=340):
                gr.HTML('<div class="ir-section-label">Job Description</div>')
                jd_input = gr.Textbox(
                    label="",
                    value=default_jd,
                    lines=18,
                    placeholder="Paste a full JD here…",
                    show_label=False,
                )
                gr.HTML('<div class="ir-section-label" style="margin-top:18px">Results Count</div>')
                top_k_slider = gr.Slider(
                    minimum=5, maximum=30, value=10, step=5,
                    label="",
                    show_label=False,
                )
                rank_button = gr.Button(
                    "⚡  Rank Candidates",
                    variant="primary",
                    size="lg",
                )

            # ── Right — Output ──────────────────────────────────
            with gr.Column(scale=2):
                gr.HTML('<div class="ir-section-label">Pipeline Status</div>')
                status_box = gr.Textbox(
                    label="",
                    interactive=False,
                    show_label=False,
                    lines=2,
                    elem_id="status-box",
                )
                gr.HTML('<div class="ir-section-label" style="margin-top:8px">Score Visualization</div>')
                breakdown_chart = gr.Plot(
                    label="",
                    show_label=False,
                )
                gr.HTML('<div class="ir-section-label" style="margin-top:8px">Ranked Results</div>')
                results_table = gr.Dataframe(
                    label="",
                    interactive=False,
                    show_label=False,
                    wrap=True,
                    max_height=580,
                )

        rank_button.click(
            fn=rank_candidates,
            inputs=[jd_input, top_k_slider],
            outputs=[results_table, status_box, breakdown_chart],
        )

        # ── Footer ──────────────────────────────────────────────
        gr.HTML(FOOTER_HTML)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the India Runs demo.")
    parser.add_argument("--smoke", action="store_true", help="Run one ranking pass without Gradio.")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    args = parser.parse_args()

    if args.smoke:
        result_df, status, _ = rank_candidates(EXAMPLE_JD, top_k=5)
        print(status)
        print(result_df[["Rank", "Candidate ID", "Current Role", "Overall"]].to_string(index=False))
        return

    demo = build_demo()
    demo.launch(server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
