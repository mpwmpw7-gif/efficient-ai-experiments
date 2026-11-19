"""Score signed relation claims separately from the two-action outcome."""
import argparse
import re

import numpy as np

from .common import ROOT, print_json, read_json


STATUS = re.compile(
    r"^(?:(?:Registry entry for )?(permit|order) ([A-Z][a-z]+) (?:is|states) "
    r"(not cancelled|cancelled|granted|denied)|The cancellation of order "
    r"([A-Z][a-z]+) has (not been|been) revoked)\.$",
    re.I,
)


def execute(spans, edges):
    """Execute each chosen governor and polarity using status statements."""
    values, restorations = {}, {}
    for span in spans:
        match = STATUS.fullmatch(span["text"])
        if match is None:
            continue
        _, name, value, revoked, polarity = match.groups()
        if revoked:
            restorations[revoked] = polarity == "been"
        else:
            values[name] = value in ("granted", "not cancelled")
    for name in values.keys() & restorations.keys():
        values[name] = restorations[name]
    return [None if edge is None or edge[0] not in values else values[edge[0]] ^ edge[1]
            for edge in edges]


def decode(candidates, scores):
    """Select the strongest positive forward margin for each action."""
    margins = scores[:, 1:3] - scores[:, [0, 3, 4]].max(axis=1, keepdims=True)
    edges = []
    for matrix in margins.reshape(2, 3, 2):
        candidate, inverse = np.unravel_index(np.argmax(matrix), matrix.shape)
        edges.append([candidates[candidate], bool(inverse)] if matrix[candidate, inverse] > 0 else None)
    return edges


def score(rows, predictions):
    if len(rows) != len(predictions) or any(r["id"] != p["id"] for r, p in zip(rows, predictions)):
        raise ValueError("Predictions must align with the held-out documents.")
    labels = np.array([r["classes"] for r in rows])
    guessed = np.array([p["pair_classes"] for p in predictions])
    reverse = labels >= 3
    true_edges = [r["graph"]["governs"] for r in rows]
    edge_correct = [p["edges"] == g for p, g in zip(predictions, true_edges)]
    state_correct = [execute(r["spans"], p["edges"]) == r["applies"] for r, p in zip(rows, predictions)]
    index = {tuple(row["id"]): i for i, row in enumerate(rows)}
    swap_pairs = [(i, index[(key[0], 1, *key[2:])]) for key, i in index.items()
                  if key[1] == 0 and (key[0], 1, *key[2:]) in index]
    return {
        "documents": len(rows),
        "forward_edges_correct": sum(e == g for p, pair in zip(predictions, true_edges)
                                     for e, g in zip(p["edges"], pair)),
        "forward_edges_total": 2 * len(rows),
        "signed_reverse_correct": int((guessed[reverse] == labels[reverse]).sum()),
        "signed_reverse_total": int(reverse.sum()),
        "complete_swaps_correct": sum(edge_correct[i] and edge_correct[j] for i, j in swap_pairs),
        "complete_swaps_total": len(swap_pairs),
        "states_correct": sum(state_correct),
        "correct_states_with_incorrect_edges": sum(s and not e for s, e in zip(state_correct, edge_correct)),
        "oracle_states_correct": sum(execute(r["spans"], g) == r["applies"] for r, g in zip(rows, true_edges)),
    }


def audit_archive(data_dir, seeds):
    rows = read_json(data_dir / "test.json.gz")
    return {str(seed): {arm: score(rows, read_json(data_dir / f"predictions_{arm}_{seed}.json.gz"))
                        for arm in ("grouped", "mixed")} for seed in seeds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=type(ROOT), default=ROOT / "data" / "relations")
    parser.add_argument("--config", default=ROOT / "configs" / "relations.json")
    args = parser.parse_args()
    print_json(audit_archive(args.data_dir, read_json(args.config)["seeds"]))


if __name__ == "__main__":
    main()
