# CogBench-RAG Project Structure

```
cogbench-rag/
├── docker/
│   └── Dockerfile
├── config/
│   └── benchmark.yaml           # Master configuration
├── src/
│   ├── __init__.py
│   ├── corpus/                   # Corpus generation
│   │   ├── __init__.py
│   │   ├── synthetic.py          # Template-based corpus builder
│   │   ├── naturalistic.py       # Wikipedia corpus loader
│   │   ├── templates.py          # Document templates per module
│   │   └── queries.py            # Query generation per module
│   ├── systems/                  # RAG system wrappers
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract base class
│   │   ├── bm25.py               # BM25 sparse retrieval
│   │   ├── dense.py              # Dense retrieval (FAISS)
│   │   ├── hybrid.py             # Hybrid sparse+dense
│   │   └── hipporag.py           # HippoRAG wrapper
│   ├── modules/                  # Benchmark modules
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract benchmark module
│   │   ├── encoding_specificity.py
│   │   ├── interference.py
│   │   ├── serial_position.py
│   │   ├── retrieval_induced_forgetting.py
│   │   ├── fan_effect.py
│   │   ├── spacing_effect.py
│   │   └── testing_effect.py
│   ├── evaluation/               # Metrics & scoring
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py  # Standard IR metrics
│   │   ├── cognitive_alignment.py # CAS computation
│   │   └── statistics.py         # Bootstrap CIs, significance tests
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── io.py
├── scripts/
│   ├── build_corpus.py           # Step 1: Generate corpus
│   ├── run_benchmark.py          # Step 2: Execute all modules
│   ├── compute_cas.py            # Step 3: Aggregate into CAS
│   └── generate_figures.py       # Step 4: Paper figures
├── tests/
│   └── ...
├── outputs/                      # Results go here
├── data/                         # Generated corpora
├── figures/                      # Generated figures
├── README.md
└── requirements.txt
```
