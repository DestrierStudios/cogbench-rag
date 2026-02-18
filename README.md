# CogBench-RAG

A cognitive benchmark for evaluating Retrieval-Augmented Generation systems against established human memory phenomena.

## Overview

CogBench-RAG tests whether RAG retrieval systems exhibit behavioral patterns documented in human memory research. It evaluates five phenomena:

| Module | Phenomenon | Source |
|--------|-----------|--------|
| Encoding Specificity | Retrieval depends on match between encoding and retrieval context | Tulving & Thomson, 1973 |
| Fan Effect | Retrieval accuracy decreases with more associated items | Anderson, 1974 |
| Interference | Prior and subsequent learning impair retrieval (proactive/retroactive) | Underwood, 1957 |
| Serial Position | Items at sequence edges are recalled better than middle items | Murdock, 1962 |
| Retrieval-Induced Forgetting | Practicing retrieval of some items suppresses related items | Anderson et al., 1994 |

Each module produces a Cognitive Alignment Score (CAS) normalized to [0, 1], where 1.0 indicates the system's behavior matches the human pattern.

## Systems Evaluated

- **BM25**: Sparse lexical retrieval (Okapi BM25, k1=1.5, b=0.75)
- **Dense Retrieval**: Semantic retrieval using all-MiniLM-L6-v2 with FAISS

## Requirements

- Docker
- Python 3.11+

## Quick Start

Build the Docker image:

```bash
docker build -t cogbench-rag -f docker/Dockerfile .
```

Run the container (Windows):

```bash
docker run -it --rm -v %cd%:/app cogbench-rag bash
```

Run the container (Mac/Linux):

```bash
docker run -it --rm -v $(pwd):/app cogbench-rag bash
```

Run the full benchmark:

```bash
PYTHONHASHSEED=0 python scripts/run_benchmark.py
```

Generate publication figures:

```bash
PYTHONHASHSEED=0 python scripts/generate_figures.py
```

## Reproducibility

All experiments require `PYTHONHASHSEED=0` and use a fixed numpy seed of 42. This ensures identical results across runs.

## Project Structure

```
cogbench-rag/
├── config/
│   └── benchmark.yaml
├── docker/
│   └── Dockerfile
├── figures/                  # Generated figures (PDF + PNG)
├── outputs/                  # Benchmark result JSON files
├── scripts/
│   ├── run_benchmark.py      # Full benchmark: 5 modules x 2 systems
│   └── generate_figures.py   # Publication-quality figures
├── src/
│   ├── modules/
│   │   ├── base.py
│   │   ├── encoding_specificity.py
│   │   ├── fan_effect.py
│   │   ├── interference.py
│   │   ├── serial_position.py
│   │   └── retrieval_induced_forgetting.py
│   └── systems/
│       ├── base.py
│       ├── bm25.py
│       └── dense.py
├── manuscript.md
├── requirements.txt
└── LICENSE
```

## Key Findings

Both BM25 and dense retrieval exhibit three human-like patterns:

- **Encoding specificity**: Both systems retrieve more accurately when query vocabulary matches document vocabulary. BM25 shows near-absolute context dependence (mismatch MRR = 0.141); dense retrieval partially bridges contextual gaps (mismatch MRR = 0.410).
- **Fan effect**: Retrieval accuracy decreases monotonically with the number of documents associated with an entity (BM25 r_s = -0.949; Dense r_s = -0.800).
- **Interference**: Retroactive interference is stronger than proactive interference in both systems, matching the well-established human asymmetry.

Neither system shows serial position effects or retrieval-induced forgetting, consistent with their stateless, order-invariant architectures.

## Citation

If you use CogBench-RAG in your research, please cite:

```
Saxena, N. (2026). Do Machines Remember Like We Do? CogBench-RAG: A Cognitive
Benchmark for Retrieval-Augmented Generation Systems. [Manuscript in preparation].
Northeastern University.
```

## License

MIT License. See [LICENSE](LICENSE) for details.
