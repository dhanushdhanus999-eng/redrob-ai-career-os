"""Phase 4 Day 21: profile retrieval-stage performance and write a report."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import build_candidate_documents, iter_job_queries, load_phase2_bundle
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.paths import DOCS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-limit", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--index-type", default="hnsw", choices=["flat", "hnsw", "ivf"])
    return parser.parse_args()


def time_it(fn, label: str) -> tuple[object, float]:
    started = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(f"{label:<30} {elapsed_ms:8.1f} ms")
    return result, elapsed_ms


def write_report(
    path: Path,
    *,
    query_job_id: str,
    candidate_count: int,
    bm25_ms: float,
    dense_ms: float,
    hybrid_ms: float,
    notes: list[str],
) -> None:
    total_ms = bm25_ms + dense_ms + hybrid_ms
    report = [
        "# Pipeline Performance Profile\n\n",
        f"- Query job ID: `{query_job_id}`\n",
        f"- Candidate pool profiled: `{candidate_count}`\n",
        f"- Dense index type: approximate retrieval benchmark\n\n",
        "| Stage | Time (ms) | Target (ms) |\n",
        "|---|---:|---:|\n",
        f"| BM25 recall | {bm25_ms:.1f} | < 50 |\n",
        f"| Dense recall | {dense_ms:.1f} | < 200 |\n",
        f"| Hybrid RRF | {hybrid_ms:.1f} | < 250 |\n",
        "| Feature extraction | N/A in this lightweight profile | < 300 |\n",
        "| LTR predict | N/A (requires trained model + labeled artifacts) | < 50 |\n",
        "| Cross-encoder | N/A in baseline profile | < 500 |\n",
        f"| Total profiled stages | {total_ms:.1f} | < 1000 |\n\n",
        "## Notes\n\n",
    ]
    for note in notes:
        report.append(f"- {note}\n")
    path.write_text("".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        bundle = load_phase2_bundle(require_labels=False)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nCopy the released jobs/candidates files into data/raw and rerun."
        ) from exc

    candidate_ids, documents = build_candidate_documents(bundle)
    if args.candidate_limit > 0:
        candidate_ids = candidate_ids[: args.candidate_limit]
        documents = documents[: args.candidate_limit]

    queries = iter_job_queries(bundle)
    if not queries:
        raise SystemExit("No job queries were found in the current dataset.")
    query_job_id, query_text = queries[0]

    print(f"Profiling query for job {query_job_id} over {len(candidate_ids)} candidates")
    bm25 = BM25Retriever()
    bm25.build_index(documents=documents, candidate_ids=candidate_ids)

    dense = DenseRetriever(
        model_name=args.dense_model,
        index_type=args.index_type,
    )
    dense.build_index(documents=documents, candidate_ids=candidate_ids)

    hybrid = HybridRetriever(bm25, dense)
    _, bm25_ms = time_it(lambda: bm25.retrieve(query_text, top_k=500), "BM25 retrieve")
    _, dense_ms = time_it(lambda: dense.retrieve(query_text, top_k=500), "Dense retrieve")
    _, hybrid_ms = time_it(
        lambda: hybrid.retrieve(query_text, top_k=args.top_k, recall_k=500),
        "Hybrid retrieve",
    )

    report_path = DOCS_DIR / "performance.md"
    notes = [
        f"Profile run used the first available job query (`{query_job_id}`).",
        "This report focuses on retrieval latency only; later pipeline stages require labeled artifacts or trained models.",
        f"Dense retrieval used `{args.dense_model}` with `{args.index_type}` indexing.",
    ]
    write_report(
        report_path,
        query_job_id=query_job_id,
        candidate_count=len(candidate_ids),
        bm25_ms=bm25_ms,
        dense_ms=dense_ms,
        hybrid_ms=hybrid_ms,
        notes=notes,
    )
    print(f"Saved performance report to: {report_path}")


if __name__ == "__main__":
    main()
