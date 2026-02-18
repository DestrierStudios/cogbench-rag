#!/usr/bin/env python3
"""
Smoke test: runs fan effect and encoding specificity modules against BM25.
Validates the full pipeline end-to-end.

Usage:
    python scripts/smoke_test.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.systems.bm25 import BM25System
from src.modules.fan_effect import FanEffectModule
from src.modules.encoding_specificity import EncodingSpecificityModule


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_fan_effect(system):
    print_header("Module: Fan Effect")

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

    print(f"\n  Effect size: {result.effect_size:.3f}")
    print(f"  Direction: {result.direction}")
    print(f"  CAS: {result.cognitive_alignment_score:.3f}")

    os.makedirs("outputs", exist_ok=True)
    module.save_results(result, "outputs")
    return result


def run_encoding_specificity(system):
    print_header("Module: Encoding Specificity")

    module_config = {
        "n_queries_per_condition": 200,
    }

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

    print(f"\n  Effect size (match - mismatch): {result.effect_size:.3f}")
    print(f"  Direction: {result.direction}")
    print(f"  CAS: {result.cognitive_alignment_score:.3f}")

    module.save_results(result, "outputs")
    return result


def main():
    np.random.seed(42)

    print_header("CogBench-RAG Smoke Test | System: BM25")

    system = BM25System({"k1": 1.5, "b": 0.75})

    r1 = run_fan_effect(system)
    r2 = run_encoding_specificity(system)

    print_header("SUMMARY")
    print(f"  Fan Effect:             CAS={r1.cognitive_alignment_score:.3f}  ({r1.direction})")
    print(f"  Encoding Specificity:   CAS={r2.cognitive_alignment_score:.3f}  ({r2.direction})")

    avg_cas = (r1.cognitive_alignment_score + r2.cognitive_alignment_score) / 2
    print(f"\n  Mean CAS across modules: {avg_cas:.3f}")
    print(f"\n{'=' * 60}")
    print("  SMOKE TEST PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
