#!/usr/bin/env python3
"""
Full benchmark: runs all 5 modules against BM25 and dense retrieval.

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
from src.modules.serial_position import SerialPositionModule
from src.modules.retrieval_induced_forgetting import RetrievalInducedForgettingModule


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_module(module, system):
    result = module.run(system)
    for cond in sorted(result.conditions.keys()):
        m = result.conditions[cond]
        mrr_str = f"MRR={m['mrr']:.3f}" if 'mrr' in m else ""
        r1_str = f"R@1={m['recall_at_1']:.3f}" if 'recall_at_1' in m else ""
        n_str = f"n={m['n_queries']}"
        print(f"  {cond:<20s}  |  {mrr_str:>10s}  {r1_str:>10s}  |  {n_str}")
    print(f"  >> Effect={result.effect_size:.3f}  Dir={result.direction}  CAS={result.cognitive_alignment_score:.3f}")
    return result


def main():
    np.random.seed(42)
    os.makedirs("outputs", exist_ok=True)

    bm25 = BM25System({"k1": 1.5, "b": 0.75})
    dense = DenseRetrievalSystem({
        "name": "dense_minilm",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 10,
    })

    module_specs = [
        ("Fan Effect", FanEffectModule({
            "fan_sizes": [1, 2, 5, 10], "n_entities_per_fan": 20,
            "n_queries_per_entity": 3, "top_k": 10,
        })),
        ("Encoding Specificity", EncodingSpecificityModule({"n_queries_per_condition": 200})),
        ("Interference", InterferenceModule({})),
        ("Serial Position", SerialPositionModule({})),
        ("Retrieval-Induced Forgetting", RetrievalInducedForgettingModule({"n_practice_rounds": 3})),
    ]

    all_results = {}

    for system in [bm25, dense]:
        print_header(f"System: {system.name}")
        for module_name, module in module_specs:
            print(f"\n  --- {module_name} ---")
            result = run_module(module, system)
            module.save_results(result, "outputs")
            all_results[(system.name, module_name)] = result

    # Comparison table
    print_header("FULL COMPARISON TABLE")
    print(f"\n  {'Module':<30s}  {'BM25 CAS':>8s}  {'Dense':>8s}  {'BM25 Dir':>12s}  {'Dense Dir':>12s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}")
    
    bm25_scores = []
    dense_scores = []
    for module_name, _ in module_specs:
        bm = all_results[("bm25", module_name)]
        dn = all_results[("dense_minilm", module_name)]
        bm25_scores.append(bm.cognitive_alignment_score)
        dense_scores.append(dn.cognitive_alignment_score)
        print(
            f"  {module_name:<30s}  "
            f"{bm.cognitive_alignment_score:>8.3f}  "
            f"{dn.cognitive_alignment_score:>8.3f}  "
            f"{bm.direction:>12s}  "
            f"{dn.direction:>12s}"
        )

    print(f"\n  {'MEAN CAS':<30s}  {np.mean(bm25_scores):>8.3f}  {np.mean(dense_scores):>8.3f}")

    # Serial position curve detail
    print_header("SERIAL POSITION CURVES")
    for sys_name in ["bm25", "dense_minilm"]:
        r = all_results[(sys_name, "Serial Position")]
        curve = r.metadata.get("position_curve", {})
        if curve:
            print(f"\n  {sys_name}:")
            positions = sorted(curve.keys())
            bars = ""
            for pos in positions:
                mrr = curve[pos]
                bar = "█" * int(mrr * 30)
                bars += f"    Pos {pos:>2d}: {bar} {mrr:.3f}\n"
            print(bars, end="")

    print(f"\n{'=' * 60}")
    print("  BENCHMARK COMPLETE — 5 modules × 2 systems")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
