"""
Stream factories for continual learning benchmarks.
Supports:
- SEA_abrupt
- SEA_recurring (A-B-A-C-A-B)
- STAGGER_recurring
- SINE_gradual
- Elec2 (real-world non-stationary)
"""
import time
from typing import Generator, Dict, Any, Optional
from river.datasets import synth, Elec2


def format_features(x: Dict[Any, Any]) -> Dict[str, float]:
    """Format feature dictionary keys to clean strings."""
    return {f"x{k}" if isinstance(k, (int, float)) else str(k): float(v) if isinstance(v, (int, float, bool)) else v 
            for k, v in x.items()}


def get_sea_recurring_stream(segment_length: int = 2500, seed: int = 42) -> Generator[Dict[str, Any], None, None]:
    """
    SEA variants in sequence A-B-A-C-A-B.
    A = variant 0, B = variant 1, C = variant 2.
    """
    concepts = [
        ("A", 0),
        ("B", 1),
        ("A", 0),
        ("C", 2),
        ("A", 0),
        ("B", 1),
    ]
    seq = 0
    for concept_label, variant in concepts:
        gen = synth.SEA(variant=variant, seed=seed + seq)
        for x, y in gen.take(segment_length):
            seq += 1
            yield {
                "seq": seq,
                "ts": time.time(),
                "x": format_features(x),
                "y": int(y),
                "stream": "SEA_recurring",
                "true_concept": concept_label,
            }


def get_sea_abrupt_stream(segment_length: int = 2500, num_segments: int = 10, seed: int = 42) -> Generator[Dict[str, Any], None, None]:
    """
    SEA abrupt stream rotating through variants 0, 1, 2, 3 every segment_length instances.
    """
    seq = 0
    for seg in range(num_segments):
        variant = seg % 4
        concept_label = f"V{variant}"
        gen = synth.SEA(variant=variant, seed=seed + seg * 1000)
        for x, y in gen.take(segment_length):
            seq += 1
            yield {
                "seq": seq,
                "ts": time.time(),
                "x": format_features(x),
                "y": int(y),
                "stream": "SEA_abrupt",
                "true_concept": concept_label,
            }


def get_stagger_recurring_stream(segment_length: int = 2000, seed: int = 42) -> Generator[Dict[str, Any], None, None]:
    """
    STAGGER recurring stream: 0 -> 1 -> 0 -> 2 -> 0 -> 1.
    """
    concepts = [
        ("C0", 0),
        ("C1", 1),
        ("C0", 0),
        ("C2", 2),
        ("C0", 0),
        ("C1", 1),
    ]
    seq = 0
    for concept_label, func in concepts:
        gen = synth.STAGGER(classification_function=func, seed=seed + seq)
        for x, y in gen.take(segment_length):
            seq += 1
            yield {
                "seq": seq,
                "ts": time.time(),
                "x": format_features(x),
                "y": int(y),
                "stream": "STAGGER_recurring",
                "true_concept": concept_label,
            }


def get_sine_gradual_stream(total_instances: int = 15000, position: int = 6000, width: int = 2000, seed: int = 42) -> Generator[Dict[str, Any], None, None]:
    """
    SINE gradual concept drift stream.
    """
    stream_a = synth.Sine(classification_function=0, seed=seed)
    stream_b = synth.Sine(classification_function=1, seed=seed + 1)
    drift_stream = synth.ConceptDriftStream(
        stream=stream_a,
        drift_stream=stream_b,
        position=position,
        width=width,
        seed=seed
    )
    seq = 0
    for x, y in drift_stream.take(total_instances):
        seq += 1
        concept_label = "S0" if seq < position else ("Transition" if seq < position + width else "S1")
        yield {
            "seq": seq,
            "ts": time.time(),
            "x": format_features(x),
            "y": int(y),
            "stream": "SINE_gradual",
            "true_concept": concept_label,
        }


def get_elec2_stream(max_instances: Optional[int] = 25000) -> Generator[Dict[str, Any], None, None]:
    """
    Elec2 real-world dataset stream.
    """
    dataset = Elec2()
    seq = 0
    for x, y in dataset:
        seq += 1
        if max_instances and seq > max_instances:
            break
        yield {
            "seq": seq,
            "ts": time.time(),
            "x": format_features(x),
            "y": int(y),
            "stream": "Elec2",
            "true_concept": "Elec2",
        }


def get_stream_generator(stream_name: str, seed: int = 42, max_instances: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
    """Factory function for streaming datasets."""
    name = stream_name.upper()
    if name in ("SEA_RECURRING", "SEA-RECURRING", "SEA_REC"):
        return get_sea_recurring_stream(segment_length=2500, seed=seed)
    elif name in ("SEA_ABRUPT", "SEA-ABRUPT"):
        return get_sea_abrupt_stream(segment_length=2500, num_segments=10, seed=seed)
    elif name in ("STAGGER_RECURRING", "STAGGER"):
        return get_stagger_recurring_stream(segment_length=2000, seed=seed)
    elif name in ("SINE_GRADUAL", "SINE"):
        return get_sine_gradual_stream(total_instances=max_instances or 15000, seed=seed)
    elif name in ("ELEC2", "ELEC"):
        return get_elec2_stream(max_instances=max_instances)
    else:
        raise ValueError(f"Unknown stream name: {stream_name}. Supported: SEA_recurring, SEA_abrupt, STAGGER_recurring, SINE_gradual, Elec2")
