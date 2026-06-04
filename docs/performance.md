# Pipeline Performance Profile

- Query job ID: `REDROB_TRACK1_MAIN_JD`
- Candidate pool profiled: `200`
- Dense index type: approximate retrieval benchmark

| Stage | Time (ms) | Target (ms) |
|---|---:|---:|
| BM25 recall | 65.2 | < 50 |
| Dense recall | 321.5 | < 200 |
| Hybrid RRF | 372.8 | < 250 |
| Feature extraction | N/A in this lightweight profile | < 300 |
| LTR predict | N/A (requires trained model + labeled artifacts) | < 50 |
| Cross-encoder | N/A in baseline profile | < 500 |
| Total profiled stages | 759.5 | < 1000 |

## Notes

- Profile run used the first available job query (`REDROB_TRACK1_MAIN_JD`).
- This report focuses on retrieval latency only; later pipeline stages require labeled artifacts or trained models.
- Dense retrieval used `BAAI/bge-small-en-v1.5` with `hnsw` indexing.
