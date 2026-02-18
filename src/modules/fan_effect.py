"""
Fan Effect Benchmark Module

Human phenomenon (Anderson, 1974):
    The more facts associated with a concept, the slower and less accurate
    retrieval of any single fact becomes. Response time increases roughly
    linearly with the number of associated facts ("fan").

RAG test:
    Create entities with varying numbers of associated documents (fan = 1, 2, 5, 10, 20).
    Query for a specific fact about each entity. Measure retrieval accuracy
    and latency as a function of fan size.

Expected human-like pattern:
    Accuracy decreases and latency increases as fan size grows.

Cognitive alignment:
    High alignment = negative correlation between fan size and accuracy
    + positive correlation between fan size and latency.
"""

import numpy as np
from scipy import stats
from collections import defaultdict
from .base import BaseBenchmarkModule, ModuleResult


class FanEffectModule(BaseBenchmarkModule):

    def __init__(self, config: dict):
        super().__init__(name="fan_effect", config=config)
        self.fan_sizes = config.get("fan_sizes", [1, 2, 5, 10, 20])
        self.n_entities_per_fan = config.get("n_entities_per_fan", 40)
        self.n_queries_per_entity = config.get("n_queries_per_entity", 5)
        self._corpus = None
        self._queries = None

    def generate_corpus(self) -> list[dict]:
        """
        Generate documents with controlled fan structure.
        Each entity has exactly `fan_size` associated documents.
        """
        documents = []
        doc_counter = 0
        entity_counter = 0

        for fan_size in self.fan_sizes:
            for _ in range(self.n_entities_per_fan):
                entity_name = f"Entity_{entity_counter:04d}"
                entity_counter += 1

                for doc_idx in range(fan_size):
                    doc_id = f"fan_doc_{doc_counter:06d}"
                    fact_id = f"fact_{doc_counter}"

                    # Each document contains a unique fact about this entity
                    content = self._generate_document(
                        entity_name, fact_id, doc_idx, fan_size
                    )

                    documents.append({
                        "doc_id": doc_id,
                        "content": content,
                        "metadata": {
                            "module": "fan_effect",
                            "entity": entity_name,
                            "fact_id": fact_id,
                            "fan_size": fan_size,
                            "doc_index_in_fan": doc_idx,
                        }
                    })
                    doc_counter += 1

        self._corpus = documents
        return documents

    def _generate_document(
        self, entity: str, fact_id: str, doc_idx: int, fan_size: int
    ) -> str:
        """
        Generate a single document about an entity.
        In production, this would use templates + LLM paraphrasing.
        This placeholder generates structured text for testing.
        """
        domains = [
            "scientific research", "economic policy", "cultural heritage",
            "technological innovation", "environmental conservation",
            "educational reform", "medical advancement", "urban development",
            "agricultural practice", "diplomatic relations", "legal framework",
            "artistic expression", "transportation infrastructure",
            "energy production", "social welfare", "military strategy",
            "financial regulation", "maritime exploration", "space technology",
            "linguistic evolution",
        ]
        domain = domains[hash(fact_id) % len(domains)]

        locations = [
            "Northern Europe", "Southeast Asia", "Central Africa",
            "South America", "Eastern Mediterranean", "Pacific Islands",
            "Central Asia", "Western Caribbean", "Southern Hemisphere",
            "Arctic Region",
        ]
        location = locations[hash(f"{fact_id}_loc") % len(locations)]

        return (
            f"{entity} is known for its contributions to {domain}. "
            f"Based in {location}, {entity} has been recognized for "
            f"achievement {fact_id} in the field of {domain}. "
            f"This accomplishment, documented as record {fact_id}, "
            f"represents a significant development in {domain} "
            f"within the {location} region. Researchers have noted that "
            f"{entity} played a central role in advancing {domain} "
            f"through the initiative referenced as {fact_id}."
        )

    def generate_queries(self) -> list[dict]:
        """
        Generate queries targeting specific facts about entities.
        Each query asks for a specific fact, and we measure whether
        the correct document is retrieved.
        """
        if self._corpus is None:
            self.generate_corpus()

        queries = []
        query_counter = 0

        # Group documents by entity
        entity_docs = defaultdict(list)
        for doc in self._corpus:
            entity = doc["metadata"]["entity"]
            entity_docs[entity].append(doc)

        for entity, docs in entity_docs.items():
            fan_size = docs[0]["metadata"]["fan_size"]

            # Sample queries targeting specific facts
            n_queries = min(self.n_queries_per_entity, len(docs))
            sampled_docs = np.random.choice(docs, size=n_queries, replace=False)

            for target_doc in sampled_docs:
                fact_id = target_doc["metadata"]["fact_id"]
                query_text = (
                    f"What is {entity}'s contribution referenced as {fact_id}?"
                )

                queries.append({
                    "query_id": f"fan_q_{query_counter:06d}",
                    "query_text": query_text,
                    "condition": f"fan_{fan_size}",
                    "expected_doc_ids": [target_doc["doc_id"]],
                    "metadata": {
                        "entity": entity,
                        "fan_size": fan_size,
                        "target_fact": fact_id,
                    },
                })
                query_counter += 1

        self._queries = queries
        return queries

    def run(self, system) -> ModuleResult:
        """Execute the fan effect benchmark on a retrieval system."""
        # Step 1: Generate corpus and queries
        corpus = self.generate_corpus()
        queries = self.generate_queries()

        # Step 2: Index corpus
        system.reset()
        system.index(corpus)

        # Step 3: Run all queries and collect results
        raw_data = []
        condition_hits = defaultdict(list)
        condition_latencies = defaultdict(list)

        for query in queries:
            result = system.retrieve(
                query_id=query["query_id"],
                query_text=query["query_text"],
                top_k=self.config.get("top_k", 10),
            )

            hit = self._compute_retrieval_hit(query, result, top_k=10)
            mrr = self._compute_mrr(query, result)
            fan_size = query["metadata"]["fan_size"]
            condition = query["condition"]

            condition_hits[condition].append(hit)
            condition_latencies[condition].append(result.latency_ms)

            raw_data.append({
                "query_id": query["query_id"],
                "condition": condition,
                "fan_size": fan_size,
                "hit_at_10": hit,
                "mrr": mrr,
                "latency_ms": result.latency_ms,
                "entity": query["metadata"]["entity"],
            })

        # Step 4: Compute condition-level metrics
        conditions = {}
        for condition in sorted(condition_hits.keys()):
            hits = condition_hits[condition]
            latencies = condition_latencies[condition]
            conditions[condition] = {
                "recall_at_10": float(np.mean(hits)),
                "mean_latency_ms": float(np.mean(latencies)),
                "n_queries": len(hits),
            }

        # Step 5: Compute cognitive alignment
        alignment = self.compute_alignment_score(conditions)

        # Compute effect size (correlation between fan size and accuracy)
        fan_sizes_arr = []
        accuracies_arr = []
        for cond, metrics in conditions.items():
            fan_size = int(cond.split("_")[1])
            fan_sizes_arr.append(fan_size)
            accuracies_arr.append(metrics["recall_at_10"])
        
        r, _ = stats.spearmanr(fan_sizes_arr, accuracies_arr)
        effect_size = float(r) if not np.isnan(r) else 0.0

        direction = "human_like" if r < -0.3 else ("opposite" if r > 0.3 else "null")

        return ModuleResult(
            module_name=self.name,
            system_name=system.name,
            conditions=conditions,
            effect_size=effect_size,
            direction=direction,
            cognitive_alignment_score=alignment,
            raw_data=raw_data,
            metadata={
                "fan_sizes": self.fan_sizes,
                "n_entities_per_fan": self.n_entities_per_fan,
                "n_total_queries": len(queries),
                "n_total_documents": len(corpus),
            },
        )

    def compute_alignment_score(self, condition_metrics: dict) -> float:
        """
        Compute cognitive alignment for the fan effect.
        
        Human pattern: accuracy decreases with fan size, latency increases.
        
        CAS = average of:
          - Normalized negative correlation (accuracy vs. fan size)
          - Normalized positive correlation (latency vs. fan size)
        
        Mapped to [0, 1] where:
          1.0 = perfect human-like pattern (strong negative acc correlation + positive latency correlation)
          0.5 = no correlation
          0.0 = perfectly opposite pattern
        """
        fan_sizes = []
        accuracies = []
        latencies = []

        for condition, metrics in sorted(condition_metrics.items()):
            fan_size = int(condition.split("_")[1])
            fan_sizes.append(fan_size)
            accuracies.append(metrics["recall_at_10"])
            latencies.append(metrics["mean_latency_ms"])

        if len(fan_sizes) < 3:
            return 0.5  # Insufficient data

        # Spearman correlations
        r_acc, _ = stats.spearmanr(fan_sizes, accuracies)
        r_lat, _ = stats.spearmanr(fan_sizes, latencies)

        # Handle NaN
        r_acc = r_acc if not np.isnan(r_acc) else 0.0
        r_lat = r_lat if not np.isnan(r_lat) else 0.0

        # Human-like: r_acc should be negative, r_lat should be positive
        # Map correlations to [0, 1]: -1 -> 1.0, 0 -> 0.5, +1 -> 0.0 (for accuracy)
        acc_alignment = (1.0 - r_acc) / 2.0  # Flipped: negative r = high alignment
        lat_alignment = (1.0 + r_lat) / 2.0  # Positive r = high alignment

        # Average the two components
        cas = (acc_alignment + lat_alignment) / 2.0
        return float(np.clip(cas, 0.0, 1.0))
