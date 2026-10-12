"""
Class-balanced Reservoir Buffer for streaming continual learning.
Provides fixed-budget memory with uniform reservoir replacement and per-class flushing.
"""
import random
from typing import Dict, List, Tuple, Any, Optional
import pympler.asizeof


class ClassBalancedReservoir:
    """
    Fixed-capacity replay buffer maintaining class balance via per-class reservoir sampling.
    Supports sample removal by class (required by AMR flush-and-refill).
    """

    def __init__(self, capacity: int = 500, seed: int = 42):
        self.capacity = max(1, capacity)
        self.seed = seed
        self.rng = random.Random(seed)
        # Store samples partitioned by class: class_id -> list of (x, y)
        self.buffers: Dict[int, List[Tuple[Dict[str, float], int]]] = {}
        # Seen counts per class for reservoir sampling
        self.seen_counts: Dict[int, int] = {}

    def _per_class_limit(self) -> int:
        num_classes = max(1, len(self.buffers))
        return max(1, self.capacity // num_classes)

    def add(self, x: Dict[str, float], y: int):
        """Add a sample (x, y) using class-balanced reservoir sampling."""
        if y not in self.buffers:
            self.buffers[y] = []
            self.seen_counts[y] = 0

        self.seen_counts[y] += 1
        class_buf = self.buffers[y]
        limit = self._per_class_limit()

        if len(class_buf) < limit:
            class_buf.append((x.copy(), y))
        else:
            # Reservoir replacement with probability limit / seen_counts[y]
            j = self.rng.randint(0, self.seen_counts[y] - 1)
            if j < limit:
                class_buf[j] = (x.copy(), y)

        # Enforce global total capacity if needed
        self._enforce_global_capacity()

    def _enforce_global_capacity(self):
        """Shrink largest buffers if total samples exceed global capacity."""
        while self.size() > self.capacity:
            # Find class with largest buffer
            largest_class = max(self.buffers.keys(), key=lambda c: len(self.buffers[c]))
            if self.buffers[largest_class]:
                self.buffers[largest_class].pop()
            else:
                break

    def sample(self, batch_size: int) -> List[Tuple[Dict[str, float], int]]:
        """Sample a batch uniformly from the combined pool of stored samples."""
        all_samples = self.get_all_samples()
        if not all_samples:
            return []
        k = min(batch_size, len(all_samples))
        return self.rng.sample(all_samples, k)

    def flush_class(self, c: int) -> int:
        """
        Removes all stored samples for class c (AMR mechanism).
        Returns the number of flushed samples.
        """
        if c in self.buffers:
            flushed_count = len(self.buffers[c])
            self.buffers[c] = []
            self.seen_counts[c] = 0
            return flushed_count
        return 0

    def get_all_samples(self) -> List[Tuple[Dict[str, float], int]]:
        samples = []
        for class_buf in self.buffers.values():
            samples.extend(class_buf)
        return samples

    def size(self) -> int:
        return sum(len(b) for b in self.buffers.values())

    def class_counts(self) -> Dict[int, int]:
        return {c: len(b) for c, b in self.buffers.items()}

    def memory_bytes(self) -> int:
        """Returns approximate memory usage in bytes using pympler."""
        return pympler.asizeof.asizeof(self.buffers)
