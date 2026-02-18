"""
Serial Position Benchmark Module

Human phenomenon (Murdock, 1962):
    Items at the beginning (primacy) and end (recency) of a sequence are
    recalled better than items in the middle, producing a U-shaped curve.

RAG test:
    Index a sequence of documents about related subtopics in a specific order.
    Query for information from documents at each position in the sequence.
    Measure retrieval accuracy as a function of indexing position.

Connection to "Lost in the Middle" (Liu et al., 2023):
    LLMs underweight information in the middle of long contexts. This module
    tests whether the RETRIEVAL stage itself introduces positional bias
    before the LLM ever sees the content — a novel upstream explanation.

Expected human-like pattern:
    U-shaped retrieval curve: positions 1-2 and N-1,N retrieved better
    than middle positions.

Prediction for systems:
    - BM25: Should show NO serial position effect (bag-of-words is order-invariant)
    - Dense (FAISS IndexFlatIP): Should show NO effect (exact search is order-invariant)
    - BUT: Systems using approximate nearest neighbors or indexing with
      insertion-order-dependent data structures MAY show effects.
    - Finding NULL here is equally publishable: it means retrieval systems
      do NOT contribute to the "lost in the middle" problem.
"""

import numpy as np
from scipy import stats
from collections import defaultdict
from .base import BaseBenchmarkModule, ModuleResult


# Each sequence is a set of documents about subtopics within a broader theme.
# All documents are deliberately similar in length, vocabulary complexity,
# and informativeness to control for content difficulty.

SEQUENCES = [
    {
        "theme": "Historical Bridges",
        "docs": [
            ("The Rialto Bridge in Venice was completed in 1591 after a design competition won by Antonio da Ponte. The single stone arch spans 48 meters across the Grand Canal and originally housed shops on both sides. Engineering challenges included driving over twelve thousand timber piles into the marshy lagoon floor to support the foundations.", "Rialto Bridge Venice 1591 Antonio da Ponte stone arch Grand Canal"),
            ("The Charles Bridge in Prague was commissioned by Emperor Charles IV in 1357 and took over fifty years to construct. Master builder Peter Parler designed the sixteen arches spanning 516 meters across the Vltava River. Thirty baroque statues were added between 1683 and 1714.", "Charles Bridge Prague Emperor Charles IV 1357 Peter Parler Vltava River baroque"),
            ("The Ponte Vecchio in Florence survived World War II because Hitler reportedly ordered its preservation. The medieval stone arch bridge built in 1345 is famous for its overhanging shops originally occupied by butchers and tanners. The Vasari Corridor runs above the shops connecting the Uffizi to the Pitti Palace.", "Ponte Vecchio Florence World War II medieval 1345 Vasari Corridor Uffizi"),
            ("The Stari Most in Mostar was originally built by Ottoman architect Mimar Hayruddin in 1566. The single arch spans 29 meters above the Neretva River at a height of 24 meters. Destroyed during the Bosnian War in 1993, it was reconstructed using original materials and techniques and reopened in 2004.", "Stari Most Mostar Ottoman Mimar Hayruddin 1566 Neretva River Bosnian War reconstructed"),
            ("The Kapellbrücke in Lucerne is Europe's oldest covered wooden bridge, originally built in 1333. The 204-meter structure crosses the Reuss River diagonally and features 111 interior paintings depicting local history. A fire in 1993 destroyed much of the bridge and most paintings, though it was rapidly rebuilt.", "Kapellbrücke Lucerne oldest covered wooden bridge 1333 Reuss River paintings fire"),
            ("The Khaju Bridge in Isfahan was constructed by Shah Abbas II around 1650 and serves simultaneously as bridge, dam, and public gathering space. Twenty-three arches span 133 meters across the Zayandeh River. Sluice gates beneath the arches regulate water flow for downstream irrigation.", "Khaju Bridge Isfahan Shah Abbas II 1650 dam arches Zayandeh River sluice gates irrigation"),
            ("The Devil's Bridge at Kromlau in Saxony was built in 1860 and deliberately designed so that its reflection in the water creates a perfect circle. The basalt stone structure arches over Rakotzsee lake in a public park. Photography of the reflection effect has made it an internationally recognized landmark.", "Devil's Bridge Kromlau Saxony 1860 reflection perfect circle basalt Rakotzsee"),
            ("The Si-o-se-pol Bridge in Isfahan contains thirty-three arches spanning 298 meters across the Zayandeh River. Built between 1599 and 1602 during the Safavid dynasty, the bridge features a lower pedestrian level with teahouses nestled between the piers. It remains one of the longest brick bridges in the world.", "Si-o-se-pol Isfahan thirty-three arches Safavid dynasty 1599 teahouses brick longest"),
            ("The Millau Viaduct in southern France holds the record as the tallest vehicular bridge, with its highest pylon reaching 343 meters. Designed by Norman Foster and Michel Virlogeux, the cable-stayed bridge spans 2.46 kilometers across the Tarn Valley. It opened in December 2004 after three years of construction.", "Millau Viaduct France tallest vehicular bridge 343 meters Norman Foster Tarn Valley 2004"),
            ("The Akashi Kaikyo Bridge in Japan has the longest central span of any suspension bridge at 1,991 meters. Connecting Kobe to Awaji Island, it withstood the 1995 Great Hanshin earthquake during construction. The bridge uses 300,000 kilometers of steel wire in its cables and was completed in 1998.", "Akashi Kaikyo Bridge Japan longest suspension span 1991 meters Kobe earthquake 1998"),
        ],
    },
    {
        "theme": "Volcanic Eruptions",
        "docs": [
            ("The eruption of Mount Vesuvius in 79 AD buried Pompeii under six meters of volcanic ash and pumice. Pliny the Younger documented the event in letters to Tacitus, providing the first detailed written account of a volcanic eruption. An estimated sixteen thousand residents died, and the city remained buried until rediscovery in 1748.", "Mount Vesuvius 79 AD Pompeii Pliny Younger Tacitus volcanic ash sixteen thousand 1748"),
            ("The 1815 eruption of Mount Tambora in Indonesia was the most powerful in recorded history, rated VEI-7. The eruption column reached 43 kilometers and ejected roughly 160 cubic kilometers of material. Global temperatures dropped by 0.5 degrees Celsius, causing 1816 to be called the Year Without a Summer.", "Mount Tambora 1815 Indonesia VEI-7 eruption column 160 cubic kilometers Year Without Summer"),
            ("Krakatoa's 1883 eruption generated tsunamis over thirty meters high that killed approximately 36,000 people across Java and Sumatra. The explosion was heard 4,800 kilometers away in Alice Springs, Australia. Atmospheric shockwaves circled the globe seven times and vivid red sunsets persisted worldwide for months.", "Krakatoa 1883 tsunami thirty meters 36000 killed Java Sumatra explosion heard shockwaves"),
            ("Mount Pinatubo in the Philippines erupted catastrophically on June 15, 1991, after 600 years of dormancy. The eruption ejected ten cubic kilometers of material and produced pyroclastic flows traveling at 100 kilometers per hour. Global temperatures decreased by approximately 0.5 degrees for two years due to sulfur aerosols.", "Mount Pinatubo Philippines 1991 pyroclastic flows sulfur aerosols global temperature decrease dormancy"),
            ("The eruption of Eyjafjallajökull in Iceland in April 2010 produced an ash cloud that closed European airspace for six days, stranding ten million passengers. The relatively small eruption was amplified by the interaction of magma with glacial meltwater. Economic losses exceeded 1.7 billion dollars for the airline industry alone.", "Eyjafjallajökull Iceland 2010 ash cloud European airspace closed passengers glacial meltwater airline losses"),
            ("Mount Saint Helens in Washington State erupted on May 18, 1980, after a magnitude 5.1 earthquake triggered the largest recorded landslide in history. The lateral blast devastated 600 square kilometers of forest. Fifty-seven people died including volcanologist David Johnston, and the mountain lost 400 meters of elevation.", "Mount Saint Helens 1980 Washington earthquake landslide lateral blast forest David Johnston elevation"),
            ("The Minoan eruption of Thera around 1600 BC may have contributed to the decline of the Minoan civilization on nearby Crete. Caldera collapse produced tsunamis estimated at 35 meters that struck the northern coast of Crete within thirty minutes. Some scholars link the event to the legend of Atlantis.", "Thera Minoan eruption 1600 BC Crete caldera tsunami Atlantis civilization decline"),
            ("Nevado del Ruiz in Colombia erupted on November 13, 1985, melting glacial ice that triggered massive lahars flowing down river valleys at sixty kilometers per hour. The town of Armero was buried under five meters of mud, killing approximately 23,000 of its 29,000 inhabitants. It became the deadliest lahar disaster in recorded history.", "Nevado del Ruiz Colombia 1985 lahars glacial ice Armero mud 23000 killed deadliest"),
            ("The ongoing eruption of Kilauea in Hawaii has been nearly continuous since 1983, making it one of the longest-running eruptions on Earth. Lava flows have added over 200 hectares of new land to the island. In 2018, a particularly destructive phase destroyed over 700 homes in the Leilani Estates subdivision.", "Kilauea Hawaii continuous eruption 1983 lava flows new land 2018 Leilani Estates homes destroyed"),
            ("Mount Pelée on Martinique erupted on May 8, 1902, producing a pyroclastic surge that destroyed the city of Saint-Pierre in under two minutes. An estimated 29,000 residents perished, with only two survivors in the city proper. The disaster led to major advances in volcanology and disaster preparedness protocols.", "Mount Pelée Martinique 1902 pyroclastic surge Saint-Pierre 29000 killed survivors volcanology"),
        ],
    },
]


class SerialPositionModule(BaseBenchmarkModule):

    def __init__(self, config: dict):
        super().__init__(name="serial_position", config=config)
        self._corpus = None
        self._queries = None

    def generate_corpus(self) -> list[dict]:
        """Generate documents preserving their sequence position metadata."""
        documents = []

        for seq_idx, sequence in enumerate(SEQUENCES):
            n_docs = len(sequence["docs"])
            for pos, (content, _query_terms) in enumerate(sequence["docs"]):
                doc_id = f"sp_seq{seq_idx}_pos{pos:02d}"
                documents.append({
                    "doc_id": doc_id,
                    "content": content,
                    "metadata": {
                        "module": "serial_position",
                        "sequence_idx": seq_idx,
                        "theme": sequence["theme"],
                        "position": pos,
                        "sequence_length": n_docs,
                        "position_zone": self._classify_position(pos, n_docs),
                    },
                })

        self._corpus = documents
        return documents

    def _classify_position(self, pos: int, seq_len: int) -> str:
        """Classify position into primacy/middle/recency zones."""
        if pos <= 1:
            return "primacy"
        elif pos >= seq_len - 2:
            return "recency"
        else:
            return "middle"

    def generate_queries(self) -> list[dict]:
        """
        Generate queries targeting specific documents at each position.
        Each query uses distinctive terms from the target document.
        """
        if self._corpus is None:
            self.generate_corpus()

        queries = []
        qc = 0

        for seq_idx, sequence in enumerate(SEQUENCES):
            for pos, (content, query_terms) in enumerate(sequence["docs"]):
                doc_id = f"sp_seq{seq_idx}_pos{pos:02d}"
                n_docs = len(sequence["docs"])
                zone = self._classify_position(pos, n_docs)

                queries.append({
                    "query_id": f"sp_q_{qc:04d}",
                    "query_text": query_terms,
                    "condition": zone,
                    "expected_doc_ids": [doc_id],
                    "metadata": {
                        "sequence_idx": seq_idx,
                        "position": pos,
                        "position_zone": zone,
                        "sequence_length": n_docs,
                    },
                })
                qc += 1

        self._queries = queries
        return queries

    def run(self, system) -> ModuleResult:
        """Execute the serial position benchmark."""
        corpus = self.generate_corpus()
        queries = self.generate_queries()

        # Index documents IN SEQUENCE ORDER — this is critical.
        # We index each sequence's documents in their defined order.
        system.reset()
        system.index(corpus)

        raw_data = []
        condition_mrrs = defaultdict(list)
        condition_hits = defaultdict(list)
        position_mrrs = defaultdict(list)

        for query in queries:
            result = system.retrieve(
                query_id=query["query_id"],
                query_text=query["query_text"],
                top_k=10,
            )

            mrr = self._compute_mrr(query, result)
            hit = self._compute_retrieval_hit(query, result, top_k=1)
            zone = query["condition"]
            pos = query["metadata"]["position"]

            condition_mrrs[zone].append(mrr)
            condition_hits[zone].append(hit)
            position_mrrs[pos].append(mrr)

            raw_data.append({
                "query_id": query["query_id"],
                "condition": zone,
                "position": pos,
                "mrr": mrr,
                "hit_at_1": hit,
                "latency_ms": result.latency_ms,
                "sequence_idx": query["metadata"]["sequence_idx"],
            })

        # Condition-level metrics (by zone)
        conditions = {}
        for zone in ["primacy", "middle", "recency"]:
            if zone in condition_mrrs:
                conditions[zone] = {
                    "mrr": float(np.mean(condition_mrrs[zone])),
                    "recall_at_1": float(np.mean(condition_hits[zone])),
                    "n_queries": len(condition_mrrs[zone]),
                }

        # Position-level metrics (for plotting the serial position curve)
        position_curve = {}
        for pos in sorted(position_mrrs.keys()):
            position_curve[pos] = float(np.mean(position_mrrs[pos]))

        alignment = self.compute_alignment_score(conditions)

        # Effect size: average of primacy and recency advantage over middle
        middle_mrr = conditions.get("middle", {}).get("mrr", 0)
        primacy_mrr = conditions.get("primacy", {}).get("mrr", 0)
        recency_mrr = conditions.get("recency", {}).get("mrr", 0)
        edge_advantage = ((primacy_mrr - middle_mrr) + (recency_mrr - middle_mrr)) / 2
        effect_size = float(edge_advantage)

        if effect_size > 0.05:
            direction = "human_like"
        elif effect_size < -0.05:
            direction = "opposite"
        else:
            direction = "null"

        return ModuleResult(
            module_name=self.name,
            system_name=system.name,
            conditions=conditions,
            effect_size=effect_size,
            direction=direction,
            cognitive_alignment_score=alignment,
            raw_data=raw_data,
            metadata={
                "n_sequences": len(SEQUENCES),
                "sequence_length": len(SEQUENCES[0]["docs"]),
                "position_curve": position_curve,
                "n_documents": len(corpus),
                "n_queries": len(queries),
            },
        )

    def compute_alignment_score(self, condition_metrics: dict) -> float:
        """
        CAS for serial position.

        Human pattern: primacy > middle AND recency > middle (U-shape).

        Score components:
        1. Primacy advantage (primacy - middle)
        2. Recency advantage (recency - middle)
        
        A flat curve (no positional effects) scores 0.
        A U-shape scores high.
        """
        primacy = condition_metrics.get("primacy", {}).get("mrr", 0)
        middle = condition_metrics.get("middle", {}).get("mrr", 0)
        recency = condition_metrics.get("recency", {}).get("mrr", 0)

        primacy_adv = max(0, primacy - middle)
        recency_adv = max(0, recency - middle)

        # Normalize: assume max plausible advantage is 0.3 MRR
        primacy_score = min(primacy_adv / 0.3, 1.0)
        recency_score = min(recency_adv / 0.3, 1.0)

        # Both present (true U-shape) is worth more
        if primacy_adv > 0.02 and recency_adv > 0.02:
            cas = (primacy_score + recency_score) / 2
        elif primacy_adv > 0.02 or recency_adv > 0.02:
            cas = max(primacy_score, recency_score) * 0.5  # Partial credit
        else:
            cas = 0.0

        return float(np.clip(cas, 0.0, 1.0))
