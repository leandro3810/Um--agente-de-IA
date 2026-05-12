from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrainingExample:
    question: str
    expected_answer: str


def split_dataset(dataset: Sequence[TrainingExample], train_ratio: float = 0.8) -> tuple[list[TrainingExample], list[TrainingExample]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio precisa estar entre 0 e 1")

    cut = int(len(dataset) * train_ratio)
    return list(dataset[:cut]), list(dataset[cut:])
