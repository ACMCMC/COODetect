import json
import math
from pathlib import Path

import numpy as np
import yaml


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def full_path(rel_path: str) -> Path:
    return repo_root() / rel_path


def chunk_slices(total_rows: int, chunk_parts: int):
    step = math.ceil(total_rows / chunk_parts)
    out = []
    start = 0
    while start < total_rows:
        end = min(total_rows, start + step)
        out.append((start, end))
        start = end
    return out


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stable_sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, n)
    idx = rng.choice(np.arange(n), size=k, replace=False)
    idx.sort()
    return idx
