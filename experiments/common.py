"""I/O and uncertainty helpers shared by the experiments."""
import gzip
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def grouped_difference(a, b, groups, seed=17, draws=2000):
    """Bootstrap the equal-group mean paired contrast, conditional on the fit."""
    a, b, groups = np.asarray(a), np.asarray(b), np.asarray(groups)
    if a.shape != b.shape or a.shape != groups.shape or a.ndim != 1 or not len(a):
        raise ValueError("Supply one nonempty vector of paired scores and group IDs.")
    contrast = a - b
    means = np.array([contrast[groups == group].mean() for group in np.unique(groups)])
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(len(means), size=(draws, len(means)))].mean(axis=1)
    return {
        "groups": len(means),
        "difference": float(means.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def print_json(value):
    print(json.dumps(value, indent=2, allow_nan=False))
