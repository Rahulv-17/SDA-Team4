"""
Representative Selection Strategies for Continual Learning Buffers.
Supports:
- Herding (selecting exemplars closest to the class mean centroid)
- Diversity / Farthest-first traversal
- Uniform reservoir baseline
"""
import numpy as np
from typing import List, Tuple, Dict


def compute_class_centroid(samples: List[Tuple[Dict[str, float], int]]) -> Dict[str, float]:
    """Calculate mean feature vector for samples."""
    if not samples:
        return {}
    keys = list(samples[0][0].keys())
    sums = {k: 0.0 for k in keys}
    for x, _ in samples:
        for k in keys:
            sums[k] += x.get(k, 0.0)
    n = len(samples)
    return {k: sums[k] / n for k in keys}


def euclidean_dist(x1: Dict[str, float], x2: Dict[str, float]) -> float:
    keys = set(x1.keys()).union(set(x2.keys()))
    return sum((x1.get(k, 0.0) - x2.get(k, 0.0)) ** 2 for k in keys) ** 0.5


def select_herding(samples: List[Tuple[Dict[str, float], int]], budget: int) -> List[Tuple[Dict[str, float], int]]:
    """Select samples closest to the class centroid."""
    if len(samples) <= budget:
        return list(samples)

    centroid = compute_class_centroid(samples)
    distances = [(euclidean_dist(x, centroid), (x, y)) for x, y in samples]
    distances.sort(key=lambda t: t[0])
    return [item[1] for item in distances[:budget]]


def select_diverse(samples: List[Tuple[Dict[str, float], int]], budget: int) -> List[Tuple[Dict[str, float], int]]:
    """Select diverse samples using farthest-first traversal."""
    if len(samples) <= budget:
        return list(samples)

    selected = [samples[0]]
    remaining = list(samples[1:])

    while len(selected) < budget and remaining:
        # Pick sample in remaining that maximizes min distance to already selected
        best_idx = -1
        max_min_dist = -1.0
        for i, (rx, ry) in enumerate(remaining):
            min_dist = min(euclidean_dist(rx, sx) for sx, _ in selected)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected
