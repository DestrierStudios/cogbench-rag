# CogBench-RAG Docker Environment
# Python 3.11 + all retrieval/evaluation dependencies

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Python dependencies - core
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    pandas>=2.0 \
    scipy>=1.11 \
    scikit-learn>=1.3 \
    matplotlib>=3.7 \
    seaborn>=0.12 \
    pyyaml>=6.0 \
    tqdm>=4.65 \
    jsonlines>=3.1

# Retrieval & embedding frameworks
RUN pip install --no-cache-dir \
    sentence-transformers>=2.2 \
    faiss-cpu>=1.7 \
    rank-bm25>=0.2 \
    chromadb>=0.4 \
    langchain>=0.1 \
    langchain-community>=0.0.10 \
    llama-index>=0.10

# Evaluation & statistics
RUN pip install --no-cache-dir \
    statsmodels>=0.14 \
    bootstrapped>=0.0.2 \
    pingouin>=0.5 \
    pytrec-eval-terrier>=0.5

# Experiment tracking & reproducibility
RUN pip install --no-cache-dir \
    mlflow>=2.9 \
    hydra-core>=1.3 \
    omegaconf>=2.3

# Corpus generation
RUN pip install --no-cache-dir \
    faker>=20.0 \
    jinja2>=3.1

# Copy project files
COPY . /app

# Default command
CMD ["bash"]
