#!/usr/bin/env python3
"""
Quick smoke test: runs the fan effect module against BM25.
Use this to validate your environment and the full pipeline end-to-end.

Usage:
    python scripts/smoke_test.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.systems.bm25 import BM25System
from src.modules.fan_effect import FanEffectModule


def main():
    np.random.seed(42)

    print("=" * 60)
    print("CogBench-RAG Smoke Test")
    print("Module: Fan Effect | System: BM25")
    print("=" * 60)

    # Small config for quick testing
    module_config = {
        "fan_sizes": [1, 2, 5, 10],
        "n_entities_per_fan": 10,
        "n_queries_per_entity": 3,
        "top_k": 10,
    }

    system_config = {
        "k1": 1.5,
        "b": 0.75,
    }

    # Initialize
    module = FanEffectModule(module_config)
    system = BM25System(system_config)

    print("\nGenerating corpus...")
    corpus = module.generate_corpus()
    print(f"  -> {len(corpus)} documents across fan sizes {module_config['fan_sizes']}")

    print("\nGenerating queries...")
    queries = module.generate_queries()
    print(f"  -> {len(queries)} queries")

    print("\nRunning benchmark...")
    result = module.run(system)

    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)

    for condition, metrics in sorted(result.conditions.items()):
        fan = condition.split("_")[1]
        print(
            f"  Fan={fan:>2s}  |  "
            f"Recall@10={metrics['recall_at_10']:.3f}  |  "
            f"Latency={metrics['mean_latency_ms']:.1f}ms  |  "
            f"n={metrics['n_queries']}"
        )

    print(f"\n  Effect size (Spearman r): {result.effect_size:.3f}")
    print(f"  Direction: {result.direction}")
    print(f"  Cognitive Alignment Score: {result.cognitive_alignment_score:.3f}")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    output_path = module.save_results(result, "outputs")
    print(f"\n  Results saved to: {output_path}")

    # Validation
    print("\n" + "=" * 60)
    if result.cognitive_alignment_score > 0:
        print("SMOKE TEST PASSED — pipeline runs end-to-end.")
    else:
        print("SMOKE TEST COMPLETED — check results above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
