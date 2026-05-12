from __future__ import annotations


def exact_match(expected: str, predicted: str) -> float:
    return 1.0 if expected.strip().lower() == predicted.strip().lower() else 0.0


def context_recall(relevant_contexts: list[str], retrieved_contexts: list[str]) -> float:
    if not relevant_contexts:
        return 1.0

    relevant = set(relevant_contexts)
    retrieved = set(retrieved_contexts)
    hits = len(relevant & retrieved)
    return hits / len(relevant)
