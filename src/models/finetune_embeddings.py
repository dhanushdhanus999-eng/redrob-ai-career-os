"""Fine-tune sentence-transformer embeddings on labeled challenge pairs."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def create_training_pairs(
    train_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    parsed_jds: dict,
    parsed_cands: dict,
    *,
    pos_threshold: float = 1.0,
    neg_threshold: float = 0.0,
) -> list:
    """Create triplet training examples from labeled job-candidate pairs."""
    from sentence_transformers import InputExample

    del parsed_jds, parsed_cands  # Reserved for future richer text construction.

    examples = []
    jobs_idx = jobs_df.set_index("job_id") if "job_id" in jobs_df.columns else jobs_df
    cands_idx = (
        candidates_df.set_index("candidate_id")
        if "candidate_id" in candidates_df.columns
        else candidates_df
    )

    job_groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"pos": [], "neg": []})
    for _, row in train_df.iterrows():
        relevance = float(row.get("relevance", 0.0))
        job_id = str(row["job_id"])
        candidate_id = str(row["candidate_id"])
        if relevance >= pos_threshold:
            job_groups[job_id]["pos"].append(candidate_id)
        elif relevance <= neg_threshold:
            job_groups[job_id]["neg"].append(candidate_id)

    for job_id, grouped_ids in job_groups.items():
        if not grouped_ids["pos"] or not grouped_ids["neg"]:
            continue
        try:
            job_row = jobs_idx.loc[int(job_id)] if str(job_id).isdigit() else jobs_idx.loc[job_id]
        except KeyError:
            continue

        job_text = str(job_row.get("description", "")) or str(job_row.get("title", ""))
        job_text = job_text[:1024]
        if not job_text.strip():
            continue

        for pos_candidate_id in grouped_ids["pos"][:5]:
            for neg_candidate_id in grouped_ids["neg"][:5]:
                try:
                    pos_row = (
                        cands_idx.loc[int(pos_candidate_id)]
                        if str(pos_candidate_id).isdigit()
                        else cands_idx.loc[pos_candidate_id]
                    )
                    neg_row = (
                        cands_idx.loc[int(neg_candidate_id)]
                        if str(neg_candidate_id).isdigit()
                        else cands_idx.loc[neg_candidate_id]
                    )
                except KeyError:
                    continue

                pos_text = str(pos_row.get("summary", "")) or str(pos_row.get("headline", ""))
                neg_text = str(neg_row.get("summary", "")) or str(neg_row.get("headline", ""))
                if not pos_text.strip() or not neg_text.strip():
                    continue
                examples.append(InputExample(texts=[job_text, pos_text[:1024], neg_text[:1024]]))

    return examples


def finetune(
    examples: list,
    *,
    base_model: str = "BAAI/bge-small-en-v1.5",
    output_path: str = "outputs/models/finetuned_embeddings",
    epochs: int = 3,
    batch_size: int = 16,
):
    """Fine-tune a compact embedding model with triplet loss."""
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(base_model)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.TripletLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=max(1, int(len(examples) * 0.1)),
        output_path=output_path,
        show_progress_bar=True,
    )
    return model
