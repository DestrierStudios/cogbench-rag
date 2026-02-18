"""
Retrieval-Induced Forgetting (RIF) Benchmark Module

Human phenomenon (Anderson, Bjork & Bjork, 1994):
    Practicing retrieval of some items from a category suppresses access
    to related but unpracticed items. Retrieving FRUIT-ORANGE impairs
    later recall of FRUIT-BANANA.

RAG test:
    Create document clusters (categories). Perform "retrieval practice"
    by querying for a subset of documents from a cluster. Then test
    retrieval of the REMAINING (unpracticed) documents from that cluster
    vs. documents from baseline clusters that had no practice at all.

    This tests whether the act of retrieving some documents from a topic
    area changes the system's ability to retrieve other related documents.

Expected human-like pattern:
    Unpracticed items in practiced categories < baseline items.
    (Practicing retrieval of some cluster members "suppresses" others.)

Why this matters for RAG:
    If retrieval practice affects subsequent retrieval, it means RAG
    systems may develop session-dependent biases — repeatedly retrieving
    certain documents could suppress access to related alternatives.
    Most RAG systems are stateless, so we predict NO RIF effect, but
    systems with caching, re-ranking feedback, or learned indices might.
"""

import numpy as np
from scipy import stats
from collections import defaultdict
from .base import BaseBenchmarkModule, ModuleResult


# Document clusters organized by category
CATEGORIES = {
    "ancient_civilizations": {
        "practiced": [
            {
                "doc_id": "rif_anc_01",
                "content": (
                    "The Sumerian civilization in Mesopotamia developed cuneiform "
                    "writing around 3400 BC, creating the earliest known written "
                    "records on clay tablets. The city-state of Ur reached its peak "
                    "under the Third Dynasty, constructing the Great Ziggurat as a "
                    "monumental temple platform. Sumerians invented the sexagesimal "
                    "number system still used in measuring time and angles."
                ),
                "query": "Sumerian cuneiform writing Mesopotamia clay tablets Ur ziggurat",
            },
            {
                "doc_id": "rif_anc_02",
                "content": (
                    "The Indus Valley Civilization flourished between 3300 and 1300 BC "
                    "across present-day Pakistan and northwest India. Mohenjo-daro "
                    "and Harappa featured advanced urban planning with grid-pattern "
                    "streets, covered drainage systems, and standardized fired brick "
                    "construction. The undeciphered Indus script remains one of "
                    "archaeology's greatest unsolved puzzles."
                ),
                "query": "Indus Valley Civilization Mohenjo-daro Harappa urban planning drainage",
            },
        ],
        "unpracticed": [
            {
                "doc_id": "rif_anc_03",
                "content": (
                    "The Norte Chico civilization in coastal Peru built monumental "
                    "architecture including platform mounds and sunken circular plazas "
                    "as early as 3000 BC, making it the oldest known civilization in "
                    "the Americas. The site of Caral features six large pyramids and "
                    "evidence of a complex society without ceramics or writing."
                ),
                "query": "Norte Chico civilization Peru Caral pyramids oldest Americas",
            },
            {
                "doc_id": "rif_anc_04",
                "content": (
                    "The Minoan civilization on Crete constructed elaborate palace "
                    "complexes at Knossos, Phaistos, and Malia between 2700 and "
                    "1450 BC. The palaces featured indoor plumbing, frescoed walls "
                    "depicting bull-leaping rituals, and the undeciphered Linear A "
                    "script. Minoan maritime trade networks extended throughout the "
                    "eastern Mediterranean."
                ),
                "query": "Minoan civilization Crete Knossos palace frescoes bull-leaping Linear A",
            },
        ],
    },
    "deep_sea_ecosystems": {
        "practiced": [
            {
                "doc_id": "rif_sea_01",
                "content": (
                    "Hydrothermal vent communities at mid-ocean ridges sustain entire "
                    "ecosystems through chemosynthesis rather than photosynthesis. Giant "
                    "tube worms Riftia pachyptila harbor symbiotic bacteria that oxidize "
                    "hydrogen sulfide to produce organic compounds. Water temperatures "
                    "at black smoker vents can exceed 400 degrees Celsius."
                ),
                "query": "hydrothermal vent chemosynthesis tube worms Riftia black smoker bacteria",
            },
            {
                "doc_id": "rif_sea_02",
                "content": (
                    "Whale falls create isolated deep-sea ecosystems when cetacean "
                    "carcasses sink to the abyssal plain. Decomposition progresses "
                    "through distinct stages lasting decades: mobile scavenger, "
                    "enrichment opportunist, and sulfophilic stages. Over 400 species "
                    "have been documented at whale fall sites, including at least "
                    "30 species found nowhere else."
                ),
                "query": "whale fall deep sea ecosystem decomposition abyssal scavenger sulfophilic species",
            },
        ],
        "unpracticed": [
            {
                "doc_id": "rif_sea_03",
                "content": (
                    "Cold seep communities form where methane and hydrogen sulfide "
                    "percolate through the seafloor in continental margin sediments. "
                    "Chemosynthetic mussels and clams dominate these sites, hosting "
                    "endosymbiotic bacteria in their gills. Methane-derived authigenic "
                    "carbonates build reef-like structures that persist for millennia "
                    "and support diverse invertebrate assemblages."
                ),
                "query": "cold seep methane hydrogen sulfide chemosynthetic mussels authigenic carbonate",
            },
            {
                "doc_id": "rif_sea_04",
                "content": (
                    "Abyssal plains below 4000 meters support sparse but diverse "
                    "communities of deposit-feeding holothurians, xenophyophores, "
                    "and stalked crinoids. These organisms subsist on marine snow — "
                    "a continuous rain of organic detritus from the productive surface. "
                    "Manganese nodule fields provide hard substrate habitat in otherwise "
                    "sediment-dominated environments."
                ),
                "query": "abyssal plain holothurians xenophyophores marine snow manganese nodules deep sea",
            },
        ],
    },
    "musical_instruments": {
        "practiced": [
            {
                "doc_id": "rif_mus_01",
                "content": (
                    "The theremin, invented by Léon Theremin in 1920, is played "
                    "without physical contact by moving hands near two metal antennae "
                    "controlling pitch and volume. Its eerie wavering tone became "
                    "iconic in science fiction film scores. Clara Rockmore developed "
                    "the fingering technique that transformed it from a novelty into "
                    "a legitimate concert instrument."
                ),
                "query": "theremin Léon Theremin 1920 antennae pitch Clara Rockmore concert",
            },
            {
                "doc_id": "rif_mus_02",
                "content": (
                    "The glass armonica was designed by Benjamin Franklin in 1761 "
                    "using nested glass bowls spinning on an iron spindle moistened "
                    "with water. Mozart and Beethoven composed pieces for the "
                    "instrument. Persistent rumors that playing it caused madness "
                    "led to bans in several German towns, though modern research "
                    "attributes any symptoms to lead in the glass."
                ),
                "query": "glass armonica Benjamin Franklin 1761 Mozart Beethoven madness lead glass",
            },
        ],
        "unpracticed": [
            {
                "doc_id": "rif_mus_03",
                "content": (
                    "The hurdy-gurdy is a stringed instrument producing sound by "
                    "a rosined wheel rubbing against the strings, operated by a "
                    "crank. Dating to the eleventh century, it was initially used "
                    "in church music before becoming a street musician's staple. "
                    "Modern revival in folk and experimental music has produced "
                    "electric and MIDI-equipped versions."
                ),
                "query": "hurdy-gurdy crank rosined wheel stringed medieval folk experimental electric",
            },
            {
                "doc_id": "rif_mus_04",
                "content": (
                    "The ondes Martenot, invented by Maurice Martenot in 1928, "
                    "uses a ribbon controller and a ring worn on the finger to "
                    "produce continuous gliding tones. Olivier Messiaen featured "
                    "it prominently in the Turangalîla Symphony. Only around a "
                    "hundred instruments were ever manufactured, making originals "
                    "extremely rare collector's items."
                ),
                "query": "ondes Martenot Maurice Martenot 1928 ribbon Messiaen Turangalîla rare",
            },
        ],
    },
    # BASELINE category — never practiced at all
    "architectural_styles": {
        "practiced": [],  # Empty — this is the baseline
        "unpracticed": [
            {
                "doc_id": "rif_arch_01",
                "content": (
                    "Brutalist architecture emerged in the 1950s emphasizing raw "
                    "concrete surfaces and monolithic geometric forms. Le Corbusier's "
                    "Unité d'Habitation in Marseille established the template. "
                    "The Barbican Estate in London and Habitat 67 in Montreal "
                    "exemplify the style's ambition to create self-contained urban "
                    "communities within massive sculptural structures."
                ),
                "query": "Brutalist architecture raw concrete Le Corbusier Barbican Habitat 67",
            },
            {
                "doc_id": "rif_arch_02",
                "content": (
                    "Deconstructivist architecture, theorized through Jacques Derrida's "
                    "philosophy, features fragmented forms, distorted surfaces, and "
                    "deliberate visual instability. Frank Gehry's Guggenheim Museum "
                    "Bilbao and Zaha Hadid's Vitra Fire Station pioneered the movement. "
                    "Daniel Libeskind's Jewish Museum Berlin uses angular voids to "
                    "represent absence and displacement."
                ),
                "query": "Deconstructivist architecture Derrida Gehry Guggenheim Hadid Libeskind fragmented",
            },
            {
                "doc_id": "rif_arch_03",
                "content": (
                    "Metabolist architecture originated in 1960s Japan proposing "
                    "cities as living organisms with replaceable modular components. "
                    "Kisho Kurokawa's Nakagin Capsule Tower in Tokyo stacked 140 "
                    "prefabricated pods onto two interconnected concrete shafts. "
                    "Kenzo Tange's Tokyo Bay Plan envisioned a floating city for "
                    "five million inhabitants extending across the bay."
                ),
                "query": "Metabolist architecture Japan Kurokawa Nakagin Capsule Tower Tange modular",
            },
            {
                "doc_id": "rif_arch_04",
                "content": (
                    "Parametric architecture uses computational algorithms to generate "
                    "complex organic forms impossible with traditional design methods. "
                    "Patrik Schumacher coined the term at the Architectural Association. "
                    "Beijing's Daxing International Airport by Zaha Hadid Architects "
                    "and the Heydar Aliyev Center in Baku demonstrate fluid geometries "
                    "derived from mathematical parametric equations."
                ),
                "query": "Parametric architecture computational algorithms Schumacher Daxing Heydar Aliyev",
            },
        ],
    },
}


class RetrievalInducedForgettingModule(BaseBenchmarkModule):

    def __init__(self, config: dict):
        super().__init__(name="retrieval_induced_forgetting", config=config)
        self.n_practice_rounds = config.get("n_practice_rounds", 3)
        self._corpus = None
        self._queries = None

    def generate_corpus(self) -> list[dict]:
        """Generate all documents from all categories."""
        documents = []
        for cat_name, cat in CATEGORIES.items():
            for doc_info in cat["practiced"] + cat["unpracticed"]:
                documents.append({
                    "doc_id": doc_info["doc_id"],
                    "content": doc_info["content"],
                    "metadata": {
                        "module": "retrieval_induced_forgetting",
                        "category": cat_name,
                        "role": "practiced" if doc_info in cat["practiced"] else "unpracticed",
                    },
                })
        self._corpus = documents
        return documents

    def generate_queries(self) -> list[dict]:
        """Generate practice queries and test queries."""
        if self._corpus is None:
            self.generate_corpus()

        queries = []
        qc = 0

        # PRACTICE phase queries — for practiced items in practiced categories
        for cat_name, cat in CATEGORIES.items():
            for doc_info in cat["practiced"]:
                for _round in range(self.n_practice_rounds):
                    queries.append({
                        "query_id": f"rif_practice_{qc:04d}",
                        "query_text": doc_info["query"],
                        "condition": "practice",
                        "expected_doc_ids": [doc_info["doc_id"]],
                        "metadata": {
                            "category": cat_name,
                            "role": "practiced",
                            "phase": "practice",
                        },
                    })
                    qc += 1

        # TEST phase queries — for unpracticed items in ALL categories
        for cat_name, cat in CATEGORIES.items():
            has_practiced_items = len(cat["practiced"]) > 0

            for doc_info in cat["unpracticed"]:
                condition = "rif_test" if has_practiced_items else "baseline"
                queries.append({
                    "query_id": f"rif_test_{qc:04d}",
                    "query_text": doc_info["query"],
                    "condition": condition,
                    "expected_doc_ids": [doc_info["doc_id"]],
                    "metadata": {
                        "category": cat_name,
                        "role": "unpracticed",
                        "phase": "test",
                        "practiced_category": has_practiced_items,
                    },
                })
                qc += 1

        self._queries = queries
        return queries

    def run(self, system) -> ModuleResult:
        """
        Execute the RIF benchmark.
        
        Phase 1: Index all documents
        Phase 2: "Practice" retrieval of practiced items (multiple rounds)
        Phase 3: Test retrieval of unpracticed items
        """
        corpus = self.generate_corpus()
        queries = self.generate_queries()

        system.reset()
        system.index(corpus)

        # Separate practice and test queries
        practice_queries = [q for q in queries if q["condition"] == "practice"]
        test_queries = [q for q in queries if q["condition"] in ("rif_test", "baseline")]

        # Phase 2: Practice retrieval (run practice queries)
        practice_results = []
        for query in practice_queries:
            result = system.retrieve(
                query_id=query["query_id"],
                query_text=query["query_text"],
                top_k=10,
            )
            mrr = self._compute_mrr(query, result)
            practice_results.append({"query_id": query["query_id"], "mrr": mrr})

        # Phase 3: Test retrieval of unpracticed items
        raw_data = []
        condition_mrrs = defaultdict(list)
        condition_hits = defaultdict(list)

        for query in test_queries:
            result = system.retrieve(
                query_id=query["query_id"],
                query_text=query["query_text"],
                top_k=10,
            )

            mrr = self._compute_mrr(query, result)
            hit = self._compute_retrieval_hit(query, result, top_k=1)
            condition = query["condition"]

            condition_mrrs[condition].append(mrr)
            condition_hits[condition].append(hit)

            raw_data.append({
                "query_id": query["query_id"],
                "condition": condition,
                "category": query["metadata"]["category"],
                "mrr": mrr,
                "hit_at_1": hit,
                "latency_ms": result.latency_ms,
            })

        # Compute condition-level metrics
        conditions = {}
        for condition in ["rif_test", "baseline"]:
            if condition in condition_mrrs:
                conditions[condition] = {
                    "mrr": float(np.mean(condition_mrrs[condition])),
                    "recall_at_1": float(np.mean(condition_hits[condition])),
                    "n_queries": len(condition_mrrs[condition]),
                }

        # Practice performance
        conditions["practice"] = {
            "mrr": float(np.mean([r["mrr"] for r in practice_results])),
            "n_queries": len(practice_results),
        }

        alignment = self.compute_alignment_score(conditions)

        # Effect size: baseline - rif_test (positive = forgetting present)
        baseline_mrr = conditions.get("baseline", {}).get("mrr", 0)
        rif_mrr = conditions.get("rif_test", {}).get("mrr", 0)
        effect_size = float(baseline_mrr - rif_mrr)

        if effect_size > 0.05:
            direction = "human_like"
        elif effect_size < -0.05:
            direction = "opposite"  # Practice actually helped related items
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
                "n_categories": len(CATEGORIES),
                "n_practice_rounds": self.n_practice_rounds,
                "n_documents": len(corpus),
                "n_practice_queries": len(practice_queries),
                "n_test_queries": len(test_queries),
            },
        )

    def compute_alignment_score(self, condition_metrics: dict) -> float:
        """
        CAS for retrieval-induced forgetting.

        Human pattern: baseline > rif_test (practicing some items
        from a category suppresses unpracticed items).

        For stateless systems, we expect NULL — which is important
        to document. CAS = 0 for null, which is a valid finding.
        """
        baseline = condition_metrics.get("baseline", {}).get("mrr", 0)
        rif_test = condition_metrics.get("rif_test", {}).get("mrr", 0)

        forgetting = max(0, baseline - rif_test)

        # Normalize: max plausible forgetting ~0.3 MRR
        cas = min(forgetting / 0.3, 1.0)
        return float(np.clip(cas, 0.0, 1.0))
