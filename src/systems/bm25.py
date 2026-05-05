"""
BM25 sparse retrieval system wrapper.
"""

from rank_bm25 import BM25Okapi
from .base import BaseRetrievalSystem, RetrievalResult


class BM25System(BaseRetrievalSystem):

    def __init__(self, config: dict):
        super().__init__(name="bm25", config=config)
        self._bm25 = None
        self._documents = []
        self._tokenized_corpus = []

    def index(self, documents: list[dict]) -> None:
        self._documents = documents
        self._tokenized_corpus = [
            doc["content"].lower().split() for doc in documents
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._is_indexed = True

    def _retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            doc = self._documents[idx]
            results.append(RetrievalResult(
                doc_id=doc["doc_id"],
                score=float(scores[idx]),
                rank=rank + 1,
                content=doc["content"],
                metadata=doc.get("metadata", {}),
            ))
        return results

    def reset(self) -> None:
        self._bm25 = None
        self._documents = []
        self._tokenized_corpus = []
        self._is_indexed = False
