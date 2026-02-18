"""
Abstract base class for RAG system wrappers.
All systems under test implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    doc_id: str
    score: float
    rank: int
    content: str
    metadata: dict


@dataclass
class QueryResult:
    """Full result for a single query."""
    query_id: str
    query_text: str
    results: list[RetrievalResult]
    latency_ms: float
    system_name: str


class BaseRetrievalSystem(ABC):
    """Abstract interface for all RAG retrieval systems."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._is_indexed = False

    @abstractmethod
    def index(self, documents: list[dict]) -> None:
        """
        Index a corpus of documents.
        
        Args:
            documents: List of dicts with keys: 'doc_id', 'content', 'metadata'
        """
        pass

    @abstractmethod
    def _retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Internal retrieval implementation."""
        pass

    def retrieve(self, query_id: str, query_text: str, top_k: int = 10) -> QueryResult:
        """
        Retrieve documents for a query with timing.
        
        Args:
            query_id: Unique identifier for this query
            query_text: The query string
            top_k: Number of results to return
            
        Returns:
            QueryResult with results and latency
        """
        if not self._is_indexed:
            raise RuntimeError(f"System '{self.name}' has not been indexed yet.")

        start = time.perf_counter()
        results = self._retrieve(query_text, top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        return QueryResult(
            query_id=query_id,
            query_text=query_text,
            results=results,
            latency_ms=latency_ms,
            system_name=self.name,
        )

    @abstractmethod
    def reset(self) -> None:
        """Clear the index. Required for modules that re-index (e.g., interference)."""
        pass

    def supports_incremental_index(self) -> bool:
        """Whether this system supports adding documents after initial indexing."""
        return False

    def supports_feedback(self) -> bool:
        """Whether this system learns from retrieval outcomes (for testing effect module)."""
        return False
