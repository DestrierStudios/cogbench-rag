"""
Encoding Specificity Benchmark Module

Human phenomenon (Tulving & Thomson, 1973):
    Memory retrieval is most effective when cues at retrieval match those
    present during encoding. A weak but encoding-matched cue outperforms
    a strong but encoding-mismatched cue.

RAG test:
    Documents are written with specific contextual framing (domain, tone,
    perspective). Queries are constructed in three conditions:
    
    (a) Context-match: query uses same domain language and framing as the document
    (b) Context-mismatch: query is semantically equivalent but uses different
        domain framing (e.g., technical vs. colloquial, different field's terminology)
    (c) Unrelated control: query about entirely different content

    We measure retrieval accuracy across conditions.

Expected human-like pattern:
    Accuracy: context_match > context_mismatch >> unrelated
    The critical comparison is (a) vs (b) — same information sought,
    different contextual framing.

Prediction for systems:
    - BM25 (lexical): Should show STRONG encoding specificity (lexical mismatch = failure)
    - Dense retrieval (semantic): Should show WEAKER encoding specificity
      (semantic similarity bridges context gap)
    - This differential is a key finding for the paper.
"""

import numpy as np
from scipy import stats
from collections import defaultdict
from .base import BaseBenchmarkModule, ModuleResult


# Each topic has a "fact" expressed in two different contextual framings
TOPIC_PAIRS = [
    {
        "topic_id": "t001",
        "core_fact": "reducing carbon emissions through plant-based diets",
        "context_a": {
            "domain": "environmental_science",
            "document": (
                "Agricultural methane contributes significantly to greenhouse gas "
                "concentrations. Livestock farming accounts for approximately 14.5% "
                "of global anthropogenic emissions according to FAO estimates. "
                "Transitioning toward plant-based dietary patterns could reduce "
                "food-related carbon footprints by up to 73%. Life cycle assessments "
                "demonstrate that legume protein production generates roughly "
                "one-fortieth the CO2 equivalent of beef per kilogram of protein."
            ),
            "query_match": "greenhouse gas reduction from shifting away from livestock agriculture",
            "query_mismatch": "health benefits of vegetarian and vegan meal plans",
        },
        "context_b": {
            "domain": "nutrition_health",
            "document": (
                "Dietitians increasingly recommend plant-forward eating patterns "
                "for both personal wellness and planetary health. Clinical studies "
                "show that reducing red meat consumption lowers cardiovascular risk "
                "while simultaneously decreasing one's environmental footprint. "
                "The Academy of Nutrition and Dietetics notes that well-planned "
                "vegetarian diets are nutritionally adequate and may provide health "
                "benefits in the prevention of certain chronic diseases."
            ),
            "query_match": "nutritional recommendations for plant-forward eating and wellness",
            "query_mismatch": "atmospheric methane from ruminant livestock farming systems",
        },
    },
    {
        "topic_id": "t002",
        "core_fact": "sleep deprivation impairs cognitive performance",
        "context_a": {
            "domain": "neuroscience",
            "document": (
                "Chronic sleep restriction disrupts prefrontal cortex functioning "
                "and impairs executive control processes. Neuroimaging studies reveal "
                "decreased activation in dorsolateral prefrontal regions following "
                "24 hours of sustained wakefulness. Adenosine accumulation in the "
                "basal forebrain drives homeostatic sleep pressure, while circadian "
                "rhythm disruption compounds the cognitive deficits observed in "
                "laboratory-controlled total sleep deprivation protocols."
            ),
            "query_match": "prefrontal cortex dysfunction from sleep restriction and wakefulness",
            "query_mismatch": "workplace productivity losses due to employee fatigue and burnout",
        },
        "context_b": {
            "domain": "occupational_health",
            "document": (
                "Employee fatigue represents a major occupational hazard costing "
                "employers an estimated $136 billion annually in lost productivity. "
                "Workers averaging fewer than six hours of sleep report significantly "
                "more errors, accidents, and absenteeism. Shift workers are "
                "particularly vulnerable, with rotating schedules disrupting natural "
                "rest patterns. Corporate wellness programs that address sleep hygiene "
                "have demonstrated measurable returns on investment."
            ),
            "query_match": "employee fatigue costs and workplace productivity impacts from poor sleep",
            "query_mismatch": "adenosine accumulation and prefrontal cortex activation during wakefulness",
        },
    },
    {
        "topic_id": "t003",
        "core_fact": "machine learning models can generate realistic images",
        "context_a": {
            "domain": "computer_science",
            "document": (
                "Diffusion models have emerged as the dominant architecture for "
                "high-fidelity image synthesis. By learning to reverse a gradual "
                "noising process, these models generate samples from complex data "
                "distributions. Classifier-free guidance enables controllable "
                "generation without auxiliary networks. Latent diffusion operates "
                "in compressed representation spaces, reducing computational "
                "requirements while maintaining perceptual quality metrics."
            ),
            "query_match": "diffusion model architectures for high-fidelity image synthesis",
            "query_mismatch": "art forgery detection and authenticating digital artwork provenance",
        },
        "context_b": {
            "domain": "art_ethics",
            "document": (
                "The proliferation of AI-generated artwork has ignited fierce debate "
                "within creative communities. Digital artists argue that generative "
                "tools trained on copyrighted works constitute unauthorized "
                "reproduction. Gallery curators face mounting challenges distinguishing "
                "human-created pieces from machine-produced imagery. The question "
                "of authorship and intellectual property in algorithmically generated "
                "visual content remains legally unresolved across most jurisdictions."
            ),
            "query_match": "AI generated art copyright disputes and creative community reactions",
            "query_mismatch": "latent diffusion model training and classifier-free guidance techniques",
        },
    },
    {
        "topic_id": "t004",
        "core_fact": "rising ocean temperatures affect marine ecosystems",
        "context_a": {
            "domain": "marine_biology",
            "document": (
                "Thermal stress triggers mass coral bleaching events when symbiotic "
                "zooxanthellae are expelled from host tissues. Sustained sea surface "
                "temperatures exceeding the local maximum monthly mean by one degree "
                "Celsius for four consecutive weeks typically initiates bleaching. "
                "Recovery depends on the magnitude and duration of thermal anomalies. "
                "Reef fish assemblages decline in diversity following severe bleaching "
                "as structural complexity of the habitat degrades."
            ),
            "query_match": "coral bleaching from thermal stress and zooxanthellae expulsion",
            "query_mismatch": "seafood industry economic losses from changing ocean conditions",
        },
        "context_b": {
            "domain": "economics",
            "document": (
                "Warming oceans threaten coastal economies dependent on fishing and "
                "tourism. The global seafood industry, valued at over $150 billion "
                "annually, faces supply chain disruptions as commercially important "
                "species migrate toward cooler waters. Island nations relying on "
                "reef tourism experience declining visitor revenues when coral "
                "degradation becomes visible. Insurance markets are recalculating "
                "coastal property risk as marine ecosystem services diminish."
            ),
            "query_match": "economic impact on fishing and tourism from warming ocean temperatures",
            "query_mismatch": "zooxanthellae thermal tolerance and reef fish assemblage changes",
        },
    },
    {
        "topic_id": "t005",
        "core_fact": "exercise improves mental health outcomes",
        "context_a": {
            "domain": "clinical_psychology",
            "document": (
                "Meta-analyses confirm that regular aerobic exercise produces "
                "antidepressant effects comparable to pharmacotherapy for mild to "
                "moderate major depressive disorder. Proposed mechanisms include "
                "increased hippocampal neurogenesis, enhanced BDNF expression, and "
                "normalization of hypothalamic-pituitary-adrenal axis dysregulation. "
                "Randomized controlled trials demonstrate dose-response relationships "
                "between exercise frequency and symptom reduction on standardized "
                "depression inventories."
            ),
            "query_match": "antidepressant effects of aerobic exercise and BDNF neurogenesis mechanisms",
            "query_mismatch": "gym membership trends and fitness app user engagement statistics",
        },
        "context_b": {
            "domain": "fitness_industry",
            "document": (
                "The global wellness market has capitalized on growing awareness "
                "that physical activity supports emotional wellbeing. Boutique "
                "fitness studios marketed as mental health boosters have seen 40% "
                "revenue growth since 2020. Wearable devices now track mood "
                "alongside step counts, and corporate gym subsidies are framed "
                "as mental health investments. The narrative connecting movement "
                "to happiness drives consumer spending across athleisure and "
                "supplement categories."
            ),
            "query_match": "wellness market growth from fitness and mental health connection marketing",
            "query_mismatch": "hippocampal neurogenesis and HPA axis normalization from aerobic exercise",
        },
    },
    {
        "topic_id": "t006",
        "core_fact": "quantum computing threatens current encryption",
        "context_a": {
            "domain": "cryptography",
            "document": (
                "Shor's algorithm enables polynomial-time factorization of large "
                "semiprimes on a sufficiently powerful quantum computer, rendering "
                "RSA and elliptic curve cryptography vulnerable. Post-quantum "
                "lattice-based schemes such as CRYSTALS-Kyber have been standardized "
                "by NIST as replacements. The transition timeline depends on "
                "achieving fault-tolerant qubit counts exceeding current noisy "
                "intermediate-scale quantum hardware by several orders of magnitude."
            ),
            "query_match": "Shor's algorithm RSA vulnerability and post-quantum lattice cryptography",
            "query_mismatch": "national security implications of foreign quantum technology programs",
        },
        "context_b": {
            "domain": "geopolitics",
            "document": (
                "Intelligence agencies worldwide are stockpiling encrypted "
                "communications under harvest-now-decrypt-later strategies, "
                "anticipating that quantum processors will eventually crack "
                "current ciphers. The race to build scalable quantum machines "
                "has become a matter of national security, with billions invested "
                "by the United States, China, and the European Union. Diplomatic "
                "tensions have risen around export controls on quantum components "
                "and talent recruitment."
            ),
            "query_match": "intelligence agencies harvest-now-decrypt-later quantum strategy geopolitics",
            "query_mismatch": "lattice-based CRYSTALS-Kyber NIST standardization post-quantum schemes",
        },
    },
    {
        "topic_id": "t007",
        "core_fact": "microplastics accumulate in food chains",
        "context_a": {
            "domain": "toxicology",
            "document": (
                "Microplastic particles below five millimeters in diameter have been "
                "detected in gastrointestinal tracts of over 800 marine species. "
                "Bioaccumulation through trophic transfer concentrates associated "
                "persistent organic pollutants including polychlorinated biphenyls "
                "and organochlorine pesticides. Histopathological examination reveals "
                "inflammatory responses in hepatic and intestinal tissues of exposed "
                "organisms at environmentally relevant concentrations."
            ),
            "query_match": "microplastic bioaccumulation trophic transfer persistent organic pollutants",
            "query_mismatch": "consumer plastic reduction campaigns and packaging industry alternatives",
        },
        "context_b": {
            "domain": "consumer_advocacy",
            "document": (
                "Environmental advocacy groups have launched campaigns urging "
                "consumers to reduce single-use plastic consumption after studies "
                "found tiny plastic fragments in drinking water, table salt, and "
                "shellfish. Major retailers are pledging to eliminate plastic "
                "packaging by 2030. Public awareness of plastics entering the food "
                "supply has driven demand for glass and metal alternatives, reshaping "
                "grocery and beverage packaging markets."
            ),
            "query_match": "consumer campaigns against single-use plastic in food and beverage packaging",
            "query_mismatch": "histopathological inflammation from polychlorinated biphenyl bioaccumulation",
        },
    },
    {
        "topic_id": "t008",
        "core_fact": "urbanization changes local climate patterns",
        "context_a": {
            "domain": "atmospheric_science",
            "document": (
                "Urban heat islands form when impervious surfaces absorb and re-emit "
                "shortwave radiation more efficiently than surrounding vegetated areas. "
                "Anthropogenic waste heat from buildings, vehicles, and industrial "
                "processes amplifies the thermal differential. Boundary layer "
                "modifications alter convective patterns, increasing afternoon "
                "thunderstorm frequency downwind of major metropolitan areas. "
                "Satellite-derived land surface temperature data confirm urban-rural "
                "gradients exceeding five degrees Celsius in summer months."
            ),
            "query_match": "urban heat island shortwave radiation boundary layer convective patterns",
            "query_mismatch": "city planning green space requirements and zoning regulations for parks",
        },
        "context_b": {
            "domain": "urban_planning",
            "document": (
                "City planners are incorporating climate-responsive design to combat "
                "rising temperatures in dense neighborhoods. Green roof mandates, "
                "reflective pavement standards, and urban tree canopy targets aim to "
                "cool streets by several degrees during peak summer. Community gardens "
                "serve dual purposes of food production and microclimate regulation. "
                "Municipal budgets increasingly allocate funds for heat mitigation "
                "as extreme heat events strain emergency services."
            ),
            "query_match": "green roof mandates reflective pavement urban cooling design strategies",
            "query_mismatch": "satellite land surface temperature urban-rural thermal gradient measurements",
        },
    },
]


class EncodingSpecificityModule(BaseBenchmarkModule):

    def __init__(self, config: dict):
        super().__init__(name="encoding_specificity", config=config)
        self.n_queries_per_condition = config.get("n_queries_per_condition", 200)
        self._corpus = None
        self._queries = None

    def generate_corpus(self) -> list[dict]:
        """Generate documents from both contextual framings of each topic."""
        documents = []

        for topic in TOPIC_PAIRS:
            for ctx_label in ["context_a", "context_b"]:
                ctx = topic[ctx_label]
                doc_id = f"es_{topic['topic_id']}_{ctx_label}"
                documents.append({
                    "doc_id": doc_id,
                    "content": ctx["document"],
                    "metadata": {
                        "module": "encoding_specificity",
                        "topic_id": topic["topic_id"],
                        "context_label": ctx_label,
                        "domain": ctx["domain"],
                        "core_fact": topic["core_fact"],
                    },
                })

        self._corpus = documents
        return documents

    def generate_queries(self) -> list[dict]:
        """
        Generate queries in three conditions per document:
        - context_match: query phrased in same domain language as document
        - context_mismatch: query seeks same info but phrased in the OTHER domain
        - unrelated: query about completely different topic (negative control)
        """
        if self._corpus is None:
            self.generate_corpus()

        queries = []
        query_counter = 0

        for topic in TOPIC_PAIRS:
            for ctx_label in ["context_a", "context_b"]:
                ctx = topic[ctx_label]
                doc_id = f"es_{topic['topic_id']}_{ctx_label}"

                # Condition 1: MATCH — query uses same domain framing
                queries.append({
                    "query_id": f"es_q_{query_counter:06d}",
                    "query_text": ctx["query_match"],
                    "condition": "context_match",
                    "expected_doc_ids": [doc_id],
                    "metadata": {
                        "topic_id": topic["topic_id"],
                        "target_context": ctx_label,
                        "target_domain": ctx["domain"],
                    },
                })
                query_counter += 1

                # Condition 2: MISMATCH — query uses OTHER domain's framing
                # (the mismatch query is stored on each context entry)
                queries.append({
                    "query_id": f"es_q_{query_counter:06d}",
                    "query_text": ctx["query_mismatch"],
                    "condition": "context_mismatch",
                    "expected_doc_ids": [doc_id],
                    "metadata": {
                        "topic_id": topic["topic_id"],
                        "target_context": ctx_label,
                        "target_domain": ctx["domain"],
                    },
                })
                query_counter += 1

        # Condition 3: UNRELATED — each document gets a query from a distant topic
        for i, topic in enumerate(TOPIC_PAIRS):
            for ctx_label in ["context_a", "context_b"]:
                doc_id = f"es_{topic['topic_id']}_{ctx_label}"
                # Pick a distant topic's query as the unrelated control
                distant_idx = (i + len(TOPIC_PAIRS) // 2) % len(TOPIC_PAIRS)
                distant_topic = TOPIC_PAIRS[distant_idx]
                distant_ctx = distant_topic["context_a"]

                queries.append({
                    "query_id": f"es_q_{query_counter:06d}",
                    "query_text": distant_ctx["query_match"],
                    "condition": "unrelated",
                    "expected_doc_ids": [doc_id],
                    "metadata": {
                        "topic_id": topic["topic_id"],
                        "target_context": ctx_label,
                        "distant_topic": distant_topic["topic_id"],
                    },
                })
                query_counter += 1

        self._queries = queries
        return queries

    def run(self, system) -> ModuleResult:
        """Execute the encoding specificity benchmark."""
        corpus = self.generate_corpus()
        queries = self.generate_queries()

        system.reset()
        system.index(corpus)

        raw_data = []
        condition_mrrs = defaultdict(list)
        condition_hits = defaultdict(list)

        for query in queries:
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
                "mrr": mrr,
                "hit_at_1": hit,
                "topic_id": query["metadata"]["topic_id"],
                "latency_ms": result.latency_ms,
            })

        # Condition-level metrics
        conditions = {}
        for condition in ["context_match", "context_mismatch", "unrelated"]:
            if condition in condition_mrrs:
                conditions[condition] = {
                    "mrr": float(np.mean(condition_mrrs[condition])),
                    "recall_at_1": float(np.mean(condition_hits[condition])),
                    "n_queries": len(condition_mrrs[condition]),
                }

        alignment = self.compute_alignment_score(conditions)

        # Effect size: MRR difference between match and mismatch
        match_mrr = conditions.get("context_match", {}).get("mrr", 0)
        mismatch_mrr = conditions.get("context_mismatch", {}).get("mrr", 0)
        effect_size = match_mrr - mismatch_mrr

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
                "n_topics": len(TOPIC_PAIRS),
                "n_documents": len(corpus),
                "n_queries": len(queries),
            },
        )

    def compute_alignment_score(self, condition_metrics: dict) -> float:
        """
        CAS for encoding specificity.

        Human pattern: match > mismatch >> unrelated
        
        Score components:
        1. Match vs mismatch gap (the encoding specificity effect itself)
        2. Correct ordering: match > mismatch > unrelated
        
        Returns [0, 1].
        """
        match = condition_metrics.get("context_match", {}).get("mrr", 0)
        mismatch = condition_metrics.get("context_mismatch", {}).get("mrr", 0)
        unrelated = condition_metrics.get("unrelated", {}).get("mrr", 0)

        # Component 1: match-mismatch gap normalized to [0, 1]
        # Max plausible gap is ~1.0, so gap/1.0 works
        gap = max(0, match - mismatch)
        gap_score = min(gap / 0.5, 1.0)  # 0.5 gap = perfect score

        # Component 2: correct ordering (match > mismatch > unrelated)
        ordering_score = 0.0
        if match > mismatch:
            ordering_score += 0.5
        if mismatch > unrelated:
            ordering_score += 0.5

        cas = (gap_score * 0.6) + (ordering_score * 0.4)
        return float(np.clip(cas, 0.0, 1.0))
