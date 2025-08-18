from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Plan:
    quotas: Dict[int, int]  # class -> number of new prompts to generate
    total_new: int


def make_plan(class_counts: Dict[int, int], target_coverage: int, max_new_per_class: int) -> Plan:
    # Balance jusqu'au max, puis applique un facteur de couverture (0..1)
    if not class_counts:
        return Plan(quotas={}, total_new=0)
    max_count = max(class_counts.values())
    raw = {cls: max(0, max_count - cnt) for cls, cnt in class_counts.items()}

    factor = max(0.0, min(1.0, float(target_coverage) / 100.0))
    quotas: Dict[int, int] = {}
    for cls, needed in raw.items():
        q = int(round(needed * factor))
        quotas[cls] = max(0, min(q, max_new_per_class))

    total_new = sum(quotas.values())
    return Plan(quotas=quotas, total_new=total_new)
