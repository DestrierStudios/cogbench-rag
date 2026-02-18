#!/usr/bin/env python3
"""
Smoke test: runs all implemented modules against BM25 and dense retrieval.

Usage:
    python scripts/smoke_test.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.systems.bm25 import BM25System
from src.systems.dense import DenseRetrievalSystem
from src.modules.fan_effect import FanEffectModule
from src.modules.encoding_specificity import EncodingSpecificityModule
from src.modules.interference import InterferenceModule


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_fan_effect(system):
    config = {
        "fan_sizes": [1, 2, 5, 10],
        "n_entities_per_fan": 20,
        "n_queries_per_entity": 3,
        "top_k": 10,
    }
    module = FanEffectModule(config)
    result = module.run(system)
    for cond, m in sorted(result.conditions.items()):
        fan = cond.split("_")[1]
        print(f"  Fan={fan:>2s}  |  MRR={m['mrr']:.3f}  |  R@1={m['recall_at_1']:.3f}  |  n={m['n_queries']}")
    print(f"  >> Effect={result.effect_size:.3f}  Dir={result.direction}  CAS={result.cognitive_alignment_score:.3f}")
    os.makedirs("outputs", exist_ok=True)
    module.save_results(result, "outputs")
    return result


def run_encoding_specificity(system):
    module = EncodingSpecificityModule({"n_queries_per_condition": 200})
    result = module.run(system)
    for cond in ["context_match", "context_mismatch", "unrelated"]:
        if cond in result.conditions:
            m = result.conditions[cond]
            label = cond.replace("_", " ").title()
            print(f"  {label:<20s}  |  MRR={m['mrr']:.3f}  |  R@1={m['recall_at_1']:.3f}  |  n={m['n_queries']}")
    print(f"  >> Effect={result.effect_size:.3f}  Dir={result.direction}  CAS={result.cognitive_alignment_score:.3f}")
    module.save_results(result, "outputs")
    return result


def run_interference(system):
    module = InterferenceModule({})
    result = module.run(system)
    for cond in ["control_phase1", "control_phase2", "ri_test", "pi_test"]:
        if cond in result.conditions:
            m = result.conditions[cond]
            labels = {
                "control_phase1": "Control (P1)",
                "control_phase2": "Control (P2)",
                "ri_test": "Retroactive Int.",
                "pi_test": "Proactive Int.",
            }
            print(f"  {labels[cond]:<20s}  |  MRR={m['mrr']:.3f}  |  R@1={m['recall_at_1']:.3f}  |  n={m['n_queries']}")
    print(f"  >> Effect={result.effect_size:.3f}  Dir={result.direction}  CAS={result.cognitive_alignment_score:.3f}")
    module.save_results(result, "outputs")
    return result


def main():
    np.random.seed(42)

    bm25 = BM25System({"k1": 1.5, "b": 0.75})
    dense = DenseRetrievalSystem({
        "name": "dense_minilm",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 10,
    })

    modules = [
        ("Fan Effect", run_fan_effect),
        ("Encoding Specificity", run_encoding_specificity),
        ("Interference", run_interference),
    ]

    all_results = {}

    for system in [bm25, dense]:
        print_header(f"System: {system.name}")
        for module_name, run_fn in modules:
            print(f"\n  --- {module_name} ---")
            result = run_fn(system)
            all_results[(system.name, module_name)] = result

    # Comparison table
    print_header("COMPARISON TABLE")
    print(f"\n  {'Module':<25s}  {'BM25 CAS':>10s}  {'Dense CAS':>10s}  {'BM25':>10s}  {'Dense':>10s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for module_name, _ in modules:
        bm = all_results[("bm25", module_name)]
        dn = all_results[("dense_minilm", module_name)]
        print(
            f"  {module_name:<25s}  "
            f"{bm.cognitive_alignment_score:>10.3f}  "
            f"{dn.cognitive_alignment_score:>10.3f}  "
            f"{bm.direction:>10s}  "
            f"{dn.direction:>10s}"
        )

    bm25_cas = np.mean([all_results[("bm25", m)].cognitive_alignment_score for m, _ in modules])
    dense_cas = np.mean([all_results[("dense_minilm", m)].cognitive_alignment_score for m, _ in modules])
    print(f"\n  BM25 mean CAS:  {bm25_cas:.3f}")
    print(f"  Dense mean CAS: {dense_cas:.3f}")
    print(f"\n{'=' * 60}")
    print("  SMOKE TEST PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
