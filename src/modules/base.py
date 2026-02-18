"""
Abstract base class for benchmark modules.
Each module tests one cognitive memory phenomenon.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import json
import os


@dataclass
class ModuleResult:
    """Results from a single module run on a single system."""
    module_name: str
    system_name: str
    conditions: dict[str, float]       # condition_name -> metric_value
    effect_size: float                  # Primary effect size for this phenomenon
    direction: str                      # "human_like", "opposite", or "null"
    cognitive_alignment_score: float    # Normalized [0, 1] alignment with human pattern
    raw_data: list[dict] = field(default_factory=list)  # Per-query results
    metadata: dict = field(default_factory=dict)


class BaseBenchmarkModule(ABC):
    """Abstract interface for all benchmark modules."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    @abstractmethod
    def generate_corpus(self) -> list[dict]:
        """
        Generate the corpus subset needed for this module.
        
        Returns:
            List of document dicts with keys: 'doc_id', 'content', 'metadata'
            metadata must include module-specific annotations.
        """
        pass

    @abstractmethod
    def generate_queries(self) -> list[dict]:
        """
        Generate the query set for this module.
        
        Returns:
            List of query dicts with keys: 'query_id', 'query_text', 'condition',
            'expected_doc_ids', and module-specific fields.
        """
        pass

    @abstractmethod
    def run(self, system) -> ModuleResult:
        """
        Execute the full benchmark module on a retrieval system.
        
        This is the main entry point. It should:
        1. Generate or load the corpus
        2. Index the corpus in the system
        3. Run all queries
        4. Compute condition-level metrics
        5. Compute the cognitive alignment score
        
        Args:
            system: A BaseRetrievalSystem instance
            
        Returns:
            ModuleResult with all metrics and raw data
        """
        pass

    @abstractmethod
    def compute_alignment_score(self, condition_metrics: dict) -> float:
        """
        Compute the cognitive alignment score for this module.
        
        Maps condition-level metrics to a [0, 1] score where:
        - 1.0 = system behavior perfectly matches human pattern
        - 0.0 = no alignment
        - Negative values = opposite of human pattern
        
        Args:
            condition_metrics: Dict of condition_name -> metric_value
            
        Returns:
            Normalized alignment score
        """
        pass

    def _compute_retrieval_hit(self, query: dict, query_result, top_k: int = 10) -> bool:
        """Check if any expected doc appears in the top-k results."""
        expected = set(query["expected_doc_ids"])
        retrieved = {r.doc_id for r in query_result.results[:top_k]}
        return bool(expected & retrieved)

    def _compute_recall_at_k(self, query: dict, query_result, k: int = 10) -> float:
        """Compute recall@k for a single query."""
        expected = set(query["expected_doc_ids"])
        retrieved = {r.doc_id for r in query_result.results[:k]}
        if not expected:
            return 0.0
        return len(expected & retrieved) / len(expected)

    def _compute_mrr(self, query: dict, query_result) -> float:
        """Compute Mean Reciprocal Rank for a single query."""
        expected = set(query["expected_doc_ids"])
        for r in query_result.results:
            if r.doc_id in expected:
                return 1.0 / r.rank
        return 0.0

    def save_results(self, result: ModuleResult, output_dir: str) -> str:
        """Save module results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.name}_{result.system_name}.json")
        with open(path, "w") as f:
            json.dump({
                "module_name": result.module_name,
                "system_name": result.system_name,
                "conditions": result.conditions,
                "effect_size": result.effect_size,
                "direction": result.direction,
                "cognitive_alignment_score": result.cognitive_alignment_score,
                "metadata": result.metadata,
                "n_raw_queries": len(result.raw_data),
            }, f, indent=2)
        return path
