from augmentor.agents.analyzer import analyze_dataset
from pathlib import Path
import json


def test_analyze_dataset(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    rows = [
        {"prompt": "Hello", "dataset": "x", "intent_disc": 0, "intent_cont": 0.1},
        {"prompt": "World", "dataset": "x", "intent_disc": 1, "intent_cont": 0.2},
    ]
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = analyze_dataset(p)
    assert res.total_rows == 2
    assert set(res.class_counts.keys()) == {0, 1}
