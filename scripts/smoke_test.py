#!/usr/bin/env python3
"""
Smoke test: runs fan effect and encoding specificity against BM25 and dense retrieval.

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


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_fan_effect(system):
    module_config = {
        "fan_sizes": [1, 2, 5, 10],
        "n_entities_per_fan": 20,
        "n_queries_per_entity": 3,
        "top_k": 10,
    }
    module = FanEffectModule(module_config)
    result = module.run(system)

    for condition, metrics in sorted(result.conditions.items()):
        fan = condition.split("_")[1]
        print(
            f"  Fan={fan:>2s}  |  "
            f"MRR={metrics['mrr']:.3f}  |  "
            f"Recall@1={metrics['recall_at_1']:.3f}  |  "
            f"n={metrics['n_queries']}"
        )

    print(f"\n  Effect size: {result.effect_size:.3f}  |  Direction: {result.direction}  |  CAS: {result.cognitive_alignment_score:.3f}")

    os.makedirs("outputs", exist_ok=True)
    module.save_results(result, "outputs")
    return result


def run_encoding_specificity(system):
    module_config = {"n_queries_per_condition": 200}
    module = EncodingSpecificityModule(module_config)
    result = module.run(system)

    for condition in ["context_match", "context_mismatch", "unrelated"]:
        if condition in result.conditions:
            metrics = result.conditions[condition]
            label = condition.replace("_", " ").title()
            print(
                f"  {label:<20s}  |  "
                f"MRR={metrics['mrr']:.3f}  |  "
                f"Recall@1={metrics['recall_at_1']:.3f}  |  "
                f"n={metrics['n_queries']}"
            )

    print(f"\n  Effect size: {result.effect_size:.3f}  |  Direction: {result.direction}  |  CAS: {result.cognitive_alignment_score:.3f}")

    module.save_results(result, "outputs")
    return result


def main():
    np.random.seed(42)

    # Initialize systems
    bm25 = BM25System({"k1": 1.5, "b": 0.75})
    dense = DenseRetrievalSystem({
        "name": "dense_minilm",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 10,
    })

    results = {}

    for system in [bm25, dense]:
        print_header(f"System: {system.name}")

        print(f"\n  --- Fan Effect ---")
        r1 = run_fan_effect(system)
        results[(system.name, "fan_effect")] = r1

        print(f"\n  --- Encoding Specificity ---")
        r2 = run_encoding_specificity(system)
        results[(system.name, "encoding_specificity")] = r2

    # Comparison table
    print_header("COMPARISON: BM25 vs Dense Retrieval")

    print(f"\n  {'Module':<25s}  {'BM25 CAS':>10s}  {'Dense CAS':>10s}  {'BM25 Dir':<12s}  {'Dense Dir':<12s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}")

    for module_name in ["fan_effect", "encoding_specificity"]:
        bm25_r = results[("bm25", module_name)]
        dense_r = results[("dense_minilm", module_name)]
        label = module_name.replace("_", " ").title()
        print(
            f"  {label:<25s}  "
            f"{bm25_r.cognitive_alignment_score:>10.3f}  "
            f"{dense_r.cognitive_alignment_score:>10.3f}  "
            f"{bm25_r.direction:<12s}  "
            f"{dense_r.direction:<12s}"
        )

    # Overall CAS per system
    print(f"\n  BM25 mean CAS:  {np.mean([results[('bm25', m)].cognitive_alignment_score for m in ['fan_effect', 'encoding_specificity']]):.3f}")
    print(f"  Dense mean CAS: {np.mean([results[('dense_minilm', m)].cognitive_alignment_score for m in ['fan_effect', 'encoding_specificity']]):.3f}")

    print(f"\n{'=' * 60}")
    print("  SMOKE TEST PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
