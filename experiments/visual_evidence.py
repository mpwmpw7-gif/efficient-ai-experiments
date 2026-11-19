"""Fit matched forward-evidence readouts and replay belief updates on archived features."""
import argparse

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

from .common import ROOT, grouped_difference, print_json, read_json


ARMS = ("original", "aligned", "unaligned")


def bits(y, p):
    return -np.log2(np.clip(np.where(y, p, 1.0 - p), 1e-6, 1.0))


def event_design(x):
    return np.column_stack((x, np.ones(len(x)))).astype(np.float64)


def fit(x, labels, dev_x, dev_labels, config):
    """Balanced logistic fit. The intercept is unregularised."""
    if labels.mean() != 0.5 or dev_labels.mean() != 0.5:
        raise ValueError("The likelihood-ratio interpretation assumes a balanced mode prior.")
    x, dev_x = event_design(x), event_design(dev_x)
    weight_penalty = 1 / (config["inverse_regularisation"] * len(x))

    def objective(w):
        z = x @ w
        base = np.logaddexp(0, z).mean() - (labels * z).mean()
        grad = x.T @ (expit(z) - labels) / len(x)
        grad[:-1] += weight_penalty * w[:-1]
        return base + weight_penalty * (w[:-1] @ w[:-1]) / 2, grad

    optimum = minimize(
        objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
        options={"maxiter": config["max_updates"], "ftol": 1e-12, "gtol": 1e-8},
    )
    dev_z = dev_x @ optimum.x
    temperature = minimize_scalar(
        lambda t: bits(dev_labels, expit(dev_z / t)).mean(),
        bounds=tuple(config["temperature_range"]), method="bounded",
    ).x
    return optimum.x.astype(np.float32), float(temperature), int(optimum.nit)


def posterior_history(llr, mask, *, deadband, switch_prior):
    """Replay the forward slot. A zero or rejected ratio leaves the prior intact."""
    updates = np.zeros(mask.shape)
    updates[mask] = np.where(np.abs(llr) <= deadband, 0.0, llr)
    history = np.full((len(mask), mask.shape[1] + 1), 0.5, dtype=np.float32)
    for step, column in enumerate(updates.T):
        history[:, step + 1] = history[:, step]
        use = column != 0
        prior = switch_prior + (1.0 - 2.0 * switch_prior) * history[use, step]
        history[use, step + 1] = expit(logit(prior) + column[use])
    return history


def outcomes(y, p):
    correct = (p >= 0.5) == y
    conf = np.maximum(p, 1 - p)
    bucket = np.minimum((10 * conf).astype(int), 9)
    ece = sum(abs(float(np.mean(conf[m] - correct[m]))) * m.mean()
              for b in range(10) if (m := (bucket == b)).any())
    return {
        "bits_per_outcome": float(bits(y, p).mean()),
        "binary_brier": float(np.square(p - y).mean()),
        "ece_10": float(ece),
        "accuracy": float(correct.mean()),
        "complete_forward_pairs": int((correct[::2] & correct[1::2]).sum()),
        "total_forward_pairs": len(y) // 2,
    }


def assess(z, gold, event_labels, neutral, mask, config):
    strength = np.abs(z)
    right = (z >= 0) == event_labels
    tape = posterior_history(z, mask, deadband=config["deadband"], switch_prior=config["switch_prior"])
    final = tape[:, -1]
    return {
        "neutral_events": int(neutral.sum()),
        "neutral_mean_abs_llr": float(strength[neutral].mean()),
        "neutral_at_least_3_to_1": int((neutral & (strength >= np.log(3))).sum()),
        "neutral_at_least_10_to_1": int((neutral & (strength >= np.log(10))).sum()),
        "identifiable_events": int((~neutral).sum()),
        "identifiable_direction_correct": int((right & ~neutral).sum()),
        "identifiable_correct_3_to_1": int((right & ~neutral & (strength >= np.log(3))).sum()),
        "event_bits": float(bits(event_labels, expit(z)).mean()),
        **outcomes(gold, final),
    }, final


def run(data_dir, config):
    models = {}
    with np.load(data_dir / "train.npz", allow_pickle=False) as train, \
            np.load(data_dir / "dev.npz", allow_pickle=False) as dev:
        for arm in ARMS:
            models[arm] = fit(train[arm], train["event_labels"], dev[arm], dev["event_labels"], config)
    summary = {"fit": {a: {"temperature": t, "updates": n, "parameters": len(w)}
                      for a, (w, t, n) in models.items()}, "splits": {}}
    for split in config["evaluation_splits"]:
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as data:
            gold, neutral, mask, event_labels = (data[k] for k in ("gold", "neutral", "mask", "event_labels"))
            scores = {"layouts": len(np.unique(data["groups"])), "methods": {}}
            predictions = {}
            for arm, (w, t, _) in models.items():
                logits = event_design(data[arm]) @ w
                raw, _ = assess(logits, gold, event_labels, neutral, mask, config)
                calibrated, predictions[arm] = assess(logits / t, gold, event_labels, neutral, mask, config)
                scores["methods"][arm] = {"raw": raw, "calibrated": calibrated}
            scores["methods"]["geometry"] = outcomes(gold, data["geometry_final"])
            scores["aligned_minus_original_bits"] = grouped_difference(
                bits(gold, predictions["aligned"]), bits(gold, predictions["original"]), data["groups"],
            )
            summary["splits"][split] = scores
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=type(ROOT), default=ROOT / "data" / "visual")
    parser.add_argument("--config", default=ROOT / "configs" / "visual.json")
    args = parser.parse_args()
    print_json(run(args.data_dir, read_json(args.config)))


if __name__ == "__main__":
    main()
