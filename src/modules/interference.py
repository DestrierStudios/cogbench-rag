"""
Interference Benchmark Module

Human phenomenon:
    Proactive interference (PI): Prior learning impairs retrieval of new information.
    Retroactive interference (RI): New learning impairs retrieval of old information.
    Both effects are stronger when the materials are more similar (Underwood, 1957).

RAG test:
    Index documents in two phases about overlapping topics:
    - Phase 1: Documents with Fact Set A about shared entities
    - Phase 2: Documents with Fact Set B about the SAME entities (updated/conflicting info)
    
    Then query for facts from each phase separately.
    
    PI: Can the system retrieve Phase 2 facts despite Phase 1 being indexed first?
    RI: Can the system still retrieve Phase 1 facts after Phase 2 was added?
    
    Control: Non-overlapping entities indexed in same phases (no interference).

Expected human-like pattern:
    - PI: Phase 2 retrieval accuracy < control (old info interferes with new)
    - RI: Phase 1 retrieval accuracy < control (new info interferes with old)
    - Higher similarity between phases = stronger interference

Prediction for systems:
    - BM25: Should show interference when Phase 1 & 2 share keywords
    - Dense: May show stronger interference since semantic similarity is higher
"""

import numpy as np
from scipy import stats
from collections import defaultdict
from .base import BaseBenchmarkModule, ModuleResult


# Shared entities that appear in BOTH phases (interference condition)
INTERFERENCE_ENTITIES = [
    {
        "entity": "Meridian Institute",
        "phase1": {
            "doc": (
                "The Meridian Institute announced a partnership with Osaka "
                "University to study tidal energy conversion in the Seto Inland "
                "Sea. Director Yuki Tanaka stated that the three-year project "
                "would deploy twelve prototype turbines by December 2024. The "
                "institute allocated a budget of forty million dollars and "
                "recruited a team of sixty oceanographic engineers."
            ),
            "query": "Meridian Institute tidal energy partnership Osaka University",
            "fact_summary": "tidal energy project with Osaka, 12 turbines, $40M budget",
        },
        "phase2": {
            "doc": (
                "The Meridian Institute has shifted focus to geothermal energy "
                "exploration in partnership with the University of Iceland. Lead "
                "researcher Sigrun Helgadottir confirmed that the institute "
                "discontinued its marine energy program to pursue drilling at "
                "three volcanic sites near Reykjanes. The revised budget stands "
                "at twenty-five million dollars with a staff of thirty-five "
                "geothermal specialists."
            ),
            "query": "Meridian Institute geothermal energy Iceland volcanic drilling",
            "fact_summary": "geothermal pivot with Iceland, discontinued marine, $25M budget",
        },
    },
    {
        "entity": "Cascade Analytics",
        "phase1": {
            "doc": (
                "Cascade Analytics released version 3.0 of its fraud detection "
                "platform, incorporating transformer-based anomaly scoring. The "
                "San Francisco startup reported processing eighteen billion "
                "transactions monthly for over two hundred banking clients. CEO "
                "Priya Venkatesh announced plans to expand into the European "
                "market by establishing offices in Frankfurt and Amsterdam."
            ),
            "query": "Cascade Analytics fraud detection platform transformer banking",
            "fact_summary": "fraud detection v3.0, 18B transactions, EU expansion",
        },
        "phase2": {
            "doc": (
                "Cascade Analytics pivoted from fraud detection to supply chain "
                "optimization after acquiring LogiFlow Systems. The company "
                "now serves fifty manufacturing clients with predictive inventory "
                "algorithms. New CTO Marcus Lindgren replaced the original "
                "leadership team and relocated headquarters to Austin, Texas. "
                "Monthly processing volume dropped to two billion records as "
                "the client base transitioned."
            ),
            "query": "Cascade Analytics supply chain optimization LogiFlow acquisition",
            "fact_summary": "supply chain pivot, acquired LogiFlow, moved to Austin",
        },
    },
    {
        "entity": "Boreal Research Station",
        "phase1": {
            "doc": (
                "The Boreal Research Station published findings linking permafrost "
                "thaw rates to methane release in subarctic Finland. Principal "
                "investigator Elina Korhonen documented a 23% acceleration in "
                "thaw depth over five years using ground-penetrating radar arrays "
                "deployed across fourteen monitoring sites near Sodankylä. The "
                "Finnish Academy of Sciences provided core funding."
            ),
            "query": "Boreal Research Station permafrost thaw methane Finland Sodankylä",
            "fact_summary": "permafrost methane study, 23% acceleration, 14 sites, Finnish funded",
        },
        "phase2": {
            "doc": (
                "The Boreal Research Station expanded operations to the Canadian "
                "Yukon, establishing a network of forty-two permafrost monitoring "
                "wells along the Dempster Highway corridor. New director James "
                "Whitehorse secured a partnership with the Geological Survey of "
                "Canada. The station's focus shifted from methane quantification "
                "to developing engineering solutions for infrastructure on "
                "degrading permafrost foundations."
            ),
            "query": "Boreal Research Station Yukon Canada permafrost infrastructure engineering",
            "fact_summary": "expanded to Yukon, 42 wells, focus shifted to infrastructure",
        },
    },
    {
        "entity": "Solaris Therapeutics",
        "phase1": {
            "doc": (
                "Solaris Therapeutics entered Phase II clinical trials for its "
                "mRNA-based melanoma vaccine candidate SLR-4401. The biotech "
                "firm enrolled 340 patients across sixteen oncology centers in "
                "the United States and United Kingdom. Chief Medical Officer "
                "Dr. Hannah Osei reported partial response rates of 38% in "
                "preliminary interim analysis."
            ),
            "query": "Solaris Therapeutics mRNA melanoma vaccine SLR-4401 clinical trial",
            "fact_summary": "melanoma vaccine Phase II, 340 patients, 38% response rate",
        },
        "phase2": {
            "doc": (
                "Solaris Therapeutics halted its melanoma program following "
                "unexpected hepatotoxicity signals in Phase II. The company "
                "redirected resources toward SLR-7802, a lipid nanoparticle "
                "therapy targeting pancreatic ductal adenocarcinoma. New "
                "partnerships with Memorial Sloan Kettering and Charité Berlin "
                "were announced for a Phase I dose-escalation study enrolling "
                "ninety patients."
            ),
            "query": "Solaris Therapeutics pancreatic cancer SLR-7802 lipid nanoparticle",
            "fact_summary": "melanoma halted, pivoted to pancreatic, SLR-7802, 90 patients",
        },
    },
    {
        "entity": "Vantage Robotics Group",
        "phase1": {
            "doc": (
                "Vantage Robotics Group demonstrated its autonomous underwater "
                "inspection drone at the Singapore Maritime Expo. The VR-Nautilus "
                "platform uses LiDAR and sonar fusion to map ship hull corrosion "
                "at depths up to two hundred meters. The company signed contracts "
                "with Maersk and CMA CGM for fleet-wide inspection services "
                "covering fourteen hundred vessels."
            ),
            "query": "Vantage Robotics underwater drone VR-Nautilus ship hull inspection",
            "fact_summary": "underwater drone VR-Nautilus, LiDAR/sonar, Maersk contract",
        },
        "phase2": {
            "doc": (
                "Vantage Robotics Group transitioned from maritime to aerospace "
                "applications following a strategic investment by Northrop Grumman. "
                "The rebranded VR-Falcon platform conducts autonomous satellite "
                "inspection in low Earth orbit. Ground operations moved from "
                "Singapore to a dedicated mission control facility in Colorado "
                "Springs. The maritime division was sold to Ocean Infinity."
            ),
            "query": "Vantage Robotics aerospace satellite inspection VR-Falcon Northrop",
            "fact_summary": "pivoted to aerospace, VR-Falcon satellite inspection, sold maritime",
        },
    },
    {
        "entity": "Helios Power Collective",
        "phase1": {
            "doc": (
                "The Helios Power Collective commissioned a 450 megawatt "
                "concentrated solar plant in the Atacama Desert of northern Chile. "
                "Using molten salt thermal storage, the facility provides eighteen "
                "hours of dispatchable electricity. Construction employed over "
                "three thousand workers and the Chilean government subsidized "
                "thirty percent of the capital cost."
            ),
            "query": "Helios Power Collective concentrated solar Atacama Chile molten salt",
            "fact_summary": "450MW solar in Atacama, molten salt storage, 18 hours dispatch",
        },
        "phase2": {
            "doc": (
                "The Helios Power Collective abandoned its Chilean solar operations "
                "after regulatory disputes over water rights for the cooling system. "
                "The collective now develops floating offshore wind platforms in the "
                "North Sea through a joint venture with Equinor. The first array "
                "of twenty turbines near Stavanger, Norway will generate 240 "
                "megawatts when operational in 2027."
            ),
            "query": "Helios Power Collective offshore wind North Sea Equinor Norway",
            "fact_summary": "abandoned Chile, pivoted to offshore wind, Equinor JV, 240MW",
        },
    },
]

# Control entities that appear in only ONE phase (no interference)
CONTROL_ENTITIES = [
    {
        "entity": "Pinnacle Acoustics Lab",
        "phase": "phase1",
        "doc": (
            "Pinnacle Acoustics Lab developed a noise cancellation algorithm "
            "that reduces urban traffic sound by 34 decibels in residential "
            "buildings. The Berlin-based laboratory tested the system across "
            "ninety-six apartments near the A100 motorway. Patent applications "
            "were filed in the European Union, Japan, and South Korea. Director "
            "Klaus Brenner presented results at the Inter-Noise conference."
        ),
        "query": "Pinnacle Acoustics noise cancellation urban traffic Berlin apartments",
        "fact_summary": "noise cancellation 34dB, 96 apartments, Berlin",
    },
    {
        "entity": "Cedarwood Genomics",
        "phase": "phase1",
        "doc": (
            "Cedarwood Genomics completed whole-genome sequencing of the bristlecone "
            "pine, the longest-lived non-clonal organism. The Vancouver laboratory "
            "identified seventy-two gene families associated with extreme longevity "
            "and stress resistance. The dataset, comprising four hundred gigabases, "
            "was deposited in NCBI GenBank and made freely available for comparative "
            "genomics research."
        ),
        "query": "Cedarwood Genomics bristlecone pine genome sequencing longevity genes",
        "fact_summary": "bristlecone pine genome, 72 longevity gene families, Vancouver",
    },
    {
        "entity": "Stratos Weather Systems",
        "phase": "phase2",
        "doc": (
            "Stratos Weather Systems launched a constellation of twelve "
            "microsatellites providing real-time precipitation nowcasting with "
            "fifteen-minute update cycles. Agricultural cooperatives in Brazil "
            "and India were early subscribers. The company processes eight "
            "terabytes of atmospheric data daily from its ground station "
            "network spanning Manaus, Hyderabad, and Nairobi."
        ),
        "query": "Stratos Weather microsatellite precipitation nowcasting agriculture",
        "fact_summary": "12 microsatellites, 15-min nowcasting, Brazil/India agriculture",
    },
    {
        "entity": "Nautilus Mineral Surveys",
        "phase": "phase2",
        "doc": (
            "Nautilus Mineral Surveys mapped polymetallic nodule deposits across "
            "thirty thousand square kilometers of the Clarion-Clipperton Zone. "
            "Using autonomous benthic crawlers equipped with X-ray fluorescence "
            "sensors, the expedition identified commercially viable concentrations "
            "of manganese, cobalt, and rare earth elements. Environmental baseline "
            "assessments documented over four hundred previously unknown species."
        ),
        "query": "Nautilus Mineral polymetallic nodules Clarion-Clipperton Zone deep sea mining",
        "fact_summary": "nodule mapping 30K sq km, benthic crawlers, 400+ new species",
    },
]


class InterferenceModule(BaseBenchmarkModule):

    def __init__(self, config: dict):
        super().__init__(name="interference", config=config)
        self._corpus = None
        self._queries = None

    def generate_corpus(self) -> list[dict]:
        """
        Generate documents in two phases.
        Phase 1 and Phase 2 documents about the same entities create interference.
        Control entities appear in only one phase.
        """
        documents = []

        # Interference entities: both phases
        for ent in INTERFERENCE_ENTITIES:
            documents.append({
                "doc_id": f"int_{ent['entity'].replace(' ', '_')}_p1",
                "content": ent["phase1"]["doc"],
                "metadata": {
                    "module": "interference",
                    "entity": ent["entity"],
                    "phase": "phase1",
                    "condition": "interference",
                    "fact_summary": ent["phase1"]["fact_summary"],
                },
            })
            documents.append({
                "doc_id": f"int_{ent['entity'].replace(' ', '_')}_p2",
                "content": ent["phase2"]["doc"],
                "metadata": {
                    "module": "interference",
                    "entity": ent["entity"],
                    "phase": "phase2",
                    "condition": "interference",
                    "fact_summary": ent["phase2"]["fact_summary"],
                },
            })

        # Control entities: single phase only
        for ent in CONTROL_ENTITIES:
            documents.append({
                "doc_id": f"int_{ent['entity'].replace(' ', '_')}_{ent['phase']}",
                "content": ent["doc"],
                "metadata": {
                    "module": "interference",
                    "entity": ent["entity"],
                    "phase": ent["phase"],
                    "condition": "control",
                    "fact_summary": ent["fact_summary"],
                },
            })

        self._corpus = documents
        return documents

    def generate_queries(self) -> list[dict]:
        """
        Generate queries that create genuine competition between phases.
        
        KEY DESIGN: Queries ask about SHARED ATTRIBUTES that changed between
        phases (budget, location, partners, focus area). Both phase documents
        are plausible matches, creating interference.
        
        Conditions:
        - ri_test: Query targets Phase 1 info using entity + Phase 1 weak cue
        - pi_test: Query targets Phase 2 info using entity + Phase 2 weak cue  
        - control_p1/p2: Query for control entities (no competing phase)
        """
        if self._corpus is None:
            self.generate_corpus()

        queries = []
        qc = 0

        # Interference queries — use entity name + weak distinguishing cue
        # The cue is enough for a human who encoded both, but creates competition
        for ent in INTERFERENCE_ENTITIES:
            entity_id = ent["entity"].replace(" ", "_")
            entity_name = ent["entity"]

            # RI test: retrieve Phase 1 facts despite Phase 2 existing
            # Use entity name + a general attribute that both phases share
            queries.append({
                "query_id": f"int_q_{qc:04d}",
                "query_text": f"{entity_name} budget staff team size project cost",
                "condition": "ri_test",
                "expected_doc_ids": [f"int_{entity_id}_p1"],
                "metadata": {
                    "entity": entity_name,
                    "target_phase": "phase1",
                    "interference_type": "retroactive",
                },
            })
            qc += 1

            # PI test: retrieve Phase 2 facts despite Phase 1 existing
            queries.append({
                "query_id": f"int_q_{qc:04d}",
                "query_text": f"{entity_name} current operations partnership collaboration",
                "condition": "pi_test",
                "expected_doc_ids": [f"int_{entity_id}_p2"],
                "metadata": {
                    "entity": entity_name,
                    "target_phase": "phase2",
                    "interference_type": "proactive",
                },
            })
            qc += 1

            # Additional RI query — different shared attribute
            queries.append({
                "query_id": f"int_q_{qc:04d}",
                "query_text": f"{entity_name} research program location headquarters",
                "condition": "ri_test",
                "expected_doc_ids": [f"int_{entity_id}_p1"],
                "metadata": {
                    "entity": entity_name,
                    "target_phase": "phase1",
                    "interference_type": "retroactive",
                },
            })
            qc += 1

            # Additional PI query
            queries.append({
                "query_id": f"int_q_{qc:04d}",
                "query_text": f"{entity_name} latest developments new direction strategy",
                "condition": "pi_test",
                "expected_doc_ids": [f"int_{entity_id}_p2"],
                "metadata": {
                    "entity": entity_name,
                    "target_phase": "phase2",
                    "interference_type": "proactive",
                },
            })
            qc += 1

        # Control queries — these entities have no competing phase
        for ent in CONTROL_ENTITIES:
            entity_id = ent["entity"].replace(" ", "_")
            entity_name = ent["entity"]
            condition = f"control_{ent['phase']}"

            # Use same generic query style as interference queries
            queries.append({
                "query_id": f"int_q_{qc:04d}",
                "query_text": f"{entity_name} budget staff team size project cost",
                "condition": condition,
                "expected_doc_ids": [f"int_{entity_id}_{ent['phase']}"],
                "metadata": {
                    "entity": entity_name,
                    "target_phase": ent["phase"],
                    "interference_type": "none",
                },
            })
            qc += 1

            queries.append({
                "query_id": f"int_q_{qc:04d}",
                "query_text": f"{entity_name} research program location headquarters",
                "condition": condition,
                "expected_doc_ids": [f"int_{entity_id}_{ent['phase']}"],
                "metadata": {
                    "entity": entity_name,
                    "target_phase": ent["phase"],
                    "interference_type": "none",
                },
            })
            qc += 1

        self._queries = queries
        return queries

    def run(self, system) -> ModuleResult:
        """Execute the interference benchmark."""
        corpus = self.generate_corpus()
        queries = self.generate_queries()

        # Index ALL documents at once (both phases)
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

            # Track whether the competing phase doc appeared instead
            entity = query["metadata"]["entity"]
            target_phase = query["metadata"]["target_phase"]
            competing_phase = "phase2" if target_phase == "phase1" else "phase1"
            competing_doc_id = f"int_{entity.replace(' ', '_')}_{competing_phase}"

            competitor_rank = None
            for r in result.results:
                if r.doc_id == competing_doc_id:
                    competitor_rank = r.rank
                    break

            raw_data.append({
                "query_id": query["query_id"],
                "condition": condition,
                "entity": entity,
                "mrr": mrr,
                "hit_at_1": hit,
                "latency_ms": result.latency_ms,
                "competitor_rank": competitor_rank,
                "top_3_doc_ids": [r.doc_id for r in result.results[:3]],
            })

        # Condition-level metrics
        conditions = {}
        for condition in sorted(condition_mrrs.keys()):
            conditions[condition] = {
                "mrr": float(np.mean(condition_mrrs[condition])),
                "recall_at_1": float(np.mean(condition_hits[condition])),
                "n_queries": len(condition_mrrs[condition]),
            }

        alignment = self.compute_alignment_score(conditions)

        # Effect size: average interference (control - test)
        control_mrr = np.mean(
            [conditions.get("control_phase1", {}).get("mrr", 0),
             conditions.get("control_phase2", {}).get("mrr", 0)]
        )
        test_mrr = np.mean(
            [conditions.get("ri_test", {}).get("mrr", 0),
             conditions.get("pi_test", {}).get("mrr", 0)]
        )
        effect_size = float(control_mrr - test_mrr)

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
                "n_interference_entities": len(INTERFERENCE_ENTITIES),
                "n_control_entities": len(CONTROL_ENTITIES),
                "n_documents": len(corpus),
                "n_queries": len(queries),
            },
        )

    def compute_alignment_score(self, condition_metrics: dict) -> float:
        """
        CAS for interference.
        
        Human pattern:
        - Control > interference tests (both PI and RI)
        - Both PI and RI present (not just one)
        
        Score: weighted combination of PI and RI presence.
        """
        control_p1 = condition_metrics.get("control_phase1", {}).get("mrr", 0)
        control_p2 = condition_metrics.get("control_phase2", {}).get("mrr", 0)
        ri_test = condition_metrics.get("ri_test", {}).get("mrr", 0)
        pi_test = condition_metrics.get("pi_test", {}).get("mrr", 0)

        avg_control = (control_p1 + control_p2) / 2 if (control_p1 + control_p2) > 0 else 0.5

        # RI effect: control_p1 - ri_test (positive = interference present)
        ri_effect = max(0, control_p1 - ri_test)
        # PI effect: control_p2 - pi_test (positive = interference present)
        pi_effect = max(0, control_p2 - pi_test)

        # Normalize: assume max plausible interference is 0.5 MRR drop
        ri_score = min(ri_effect / 0.5, 1.0)
        pi_score = min(pi_effect / 0.5, 1.0)

        # Both types present is more human-like than just one
        both_present_bonus = 0.2 if (ri_effect > 0.02 and pi_effect > 0.02) else 0.0

        cas = (ri_score * 0.4) + (pi_score * 0.4) + both_present_bonus
        return float(np.clip(cas, 0.0, 1.0))
