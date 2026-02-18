"""
Dense retrieval system using sentence-transformers + FAISS.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .base import BaseRetrievalSystem, RetrievalResult


class DenseRetrievalSystem(BaseRetrievalSystem):

    def __init__(self, config: dict):
        name = config.get("name", "dense")
        super().__init__(name=name, config=config)
        self._model = SentenceTransformer(config["embedding_model"])
        self._index = None
        self._documents = []
        self._dimension = None

    def index(self, documents: list[dict]) -> None:
        self._documents = documents
        contents = [doc["content"] for doc in documents]

        # Encode all documents
        embeddings = self._model.encode(
            contents, show_progress_bar=True, batch_size=64, normalize_embeddings=True
        )
        self._dimension = embeddings.shape[1]

        # Build FAISS index (inner product for normalized vectors = cosine similarity)
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(embeddings.astype(np.float32))
        self._is_indexed = True

    def _retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        query_embedding = self._model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self._index.search(query_embedding, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                continue
            doc = self._documents[idx]
            results.append(RetrievalResult(
                doc_id=doc["doc_id"],
                score=float(score),
                rank=rank + 1,
                content=doc["content"],
                metadata=doc.get("metadata", {}),
            ))
        return results

    def reset(self) -> None:
        self._index = None
        self._documents = []
        self._is_indexed = False
