# Methodology Notes

Initial ideas worth trying once the dataset is available:

1. Use a multi-stage pipeline: hybrid recall, feature generation, learning to
   rank, then optional reranking.
2. Treat behavioral and freshness signals as a first-class feature group because
   the brief explicitly highlights them.
3. Build structured representations of jobs and candidates so we can score
   must-have skills, seniority, and experience alignment separately from plain
   text similarity.
4. Keep evaluation deterministic from the start with fixed job-level splits and
   shared scoring utilities.
5. Document every experiment and feature group so later phases can support an
   ablation study and strong README storytelling.
