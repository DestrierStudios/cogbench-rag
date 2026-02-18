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
        Generate a single document about an entity with high lexical diversity.
        Each document uses a unique combination of domain, action verbs,
        descriptors, and proper nouns to ensure BM25 can discriminate.
        """
        # Use hash to deterministically select unique word combinations
        h = hash(fact_id)

        domains = [
            "quantum computing", "marine biology", "constitutional law",
            "volcanology", "behavioral economics", "Renaissance sculpture",
            "antibiotic resistance", "glacial erosion", "jazz improvisation",
            "orbital mechanics", "textile manufacturing", "coral reef ecology",
            "cryptographic protocols", "medieval cartography", "gene therapy",
            "earthquake prediction", "operatic composition", "tidal energy",
            "ceramic engineering", "forensic entomology",
        ]

        actions = [
            "pioneered a breakthrough in", "published landmark findings on",
            "developed a novel framework for", "secured international patents related to",
            "led a multinational consortium studying", "authored the definitive guide to",
            "established a research center dedicated to", "won the Fielding Prize for work in",
            "transformed conventional approaches to", "uncovered critical evidence regarding",
            "designed the prototype system for", "demonstrated unexpected connections in",
            "challenged prevailing theories about", "received widespread acclaim for advancing",
            "organized the first global summit on", "completed a decade-long study of",
            "introduced computational methods to", "mapped the complete structure of",
            "trained over two hundred specialists in", "discovered the underlying mechanism of",
        ]

        locations = [
            "Reykjavik", "Kuala Lumpur", "Nairobi", "Montevideo", "Thessaloniki",
            "Suva", "Tashkent", "Bridgetown", "Christchurch", "Tromsø",
            "Cartagena", "Bruges", "Hanoi", "Accra", "Ljubljana",
            "Muscat", "Quito", "Bergen", "Dakar", "Tallinn",
        ]

        years = [
            "1987", "1993", "2001", "2006", "2011",
            "1995", "2003", "2008", "2014", "2018",
            "1991", "1998", "2005", "2009", "2016",
            "1989", "1996", "2002", "2012", "2019",
        ]

        outcomes = [
            "resulting in three subsequent patents",
            "which reshaped the field for a generation",
            "attracting over forty million dollars in funding",
            "prompting regulatory changes across twelve countries",
            "earning recognition from the National Academy",
            "inspiring a wave of follow-up investigations",
            "leading to commercial applications within five years",
            "generating over two hundred peer-reviewed citations",
            "creating a standard adopted by the ISO committee",
            "establishing a new subdiscipline entirely",
            "forming the basis of current graduate curricula",
            "triggering a public debate about ethical guidelines",
            "saving an estimated fifteen thousand lives annually",
            "reducing operational costs by thirty-seven percent",
            "opening previously inaccessible archives to scholars",
            "resolving a controversy that lasted four decades",
            "connecting two previously unrelated research programs",
            "enabling precision measurements at the nanoscale",
            "producing the largest dataset of its kind",
            "fundamentally altering clinical treatment protocols",
        ]

        domain = domains[h % len(domains)]
        action = actions[(h >> 4) % len(actions)]
        location = locations[(h >> 8) % len(locations)]
        year = years[(h >> 12) % len(years)]
        outcome = outcomes[(h >> 16) % len(outcomes)]

        return (
            f"In {year}, {entity} {action} {domain} while based at "
            f"the research institute in {location}. This initiative, "
            f"catalogued as {fact_id}, focused specifically on {domain} "
            f"and involved collaboration with specialists in {location}. "
            f"The work on {fact_id} by {entity} proved significant, "
            f"{outcome}. Experts in {domain} have since cited the "
            f"{location} project as a turning point in the discipline."
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

                # FAN EFFECT DESIGN: Query uses entity name + domain keyword.
                # NOT the unique fact_id or long verbatim phrases.
                # This creates the competition that produces the fan effect:
                # - Fan=1: only 1 doc about this entity → easy, always rank 1
                # - Fan=10: 10 docs about this entity, system must use the
                #   domain keyword to rank the right one highest
                content = target_doc["content"]
                # Extract just the domain keyword (e.g., "quantum computing")
                # It appears after the action verb phrase in our template
                h = hash(fact_id)
                domains = [
                    "quantum computing", "marine biology", "constitutional law",
                    "volcanology", "behavioral economics", "Renaissance sculpture",
                    "antibiotic resistance", "glacial erosion", "jazz improvisation",
                    "orbital mechanics", "textile manufacturing", "coral reef ecology",
                    "cryptographic protocols", "medieval cartography", "gene therapy",
                    "earthquake prediction", "operatic composition", "tidal energy",
                    "ceramic engineering", "forensic entomology",
                ]
                domain = domains[h % len(domains)]

                query_text = f"{entity} {domain}"

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
        condition_mrrs = defaultdict(list)
        condition_hits_at_1 = defaultdict(list)
        condition_latencies = defaultdict(list)

        for query in queries:
            result = system.retrieve(
                query_id=query["query_id"],
                query_text=query["query_text"],
                top_k=self.config.get("top_k", 10),
            )

            mrr = self._compute_mrr(query, result)
            hit_at_1 = self._compute_retrieval_hit(query, result, top_k=1)
            fan_size = query["metadata"]["fan_size"]
            condition = query["condition"]

            condition_mrrs[condition].append(mrr)
            condition_hits_at_1[condition].append(hit_at_1)
            condition_latencies[condition].append(result.latency_ms)

            raw_data.append({
                "query_id": query["query_id"],
                "condition": condition,
                "fan_size": fan_size,
                "mrr": mrr,
                "hit_at_1": hit_at_1,
                "latency_ms": result.latency_ms,
                "entity": query["metadata"]["entity"],
                "top_3_doc_ids": [r.doc_id for r in result.results[:3]],
            })

        # Step 4: Compute condition-level metrics
        conditions = {}
        for condition in sorted(condition_mrrs.keys()):
            mrrs = condition_mrrs[condition]
            hits = condition_hits_at_1[condition]
            latencies = condition_latencies[condition]
            conditions[condition] = {
                "mrr": float(np.mean(mrrs)),
                "recall_at_1": float(np.mean(hits)),
                "mean_latency_ms": float(np.mean(latencies)),
                "n_queries": len(mrrs),
            }

        # Step 5: Compute cognitive alignment
        alignment = self.compute_alignment_score(conditions)

        # Compute effect size (correlation between fan size and MRR)
        fan_sizes_arr = []
        mrr_arr = []
        for cond, metrics in conditions.items():
            fan_size = int(cond.split("_")[1])
            fan_sizes_arr.append(fan_size)
            mrr_arr.append(metrics["mrr"])
        
        r, _ = stats.spearmanr(fan_sizes_arr, mrr_arr)
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
            accuracies.append(metrics["mrr"])
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
