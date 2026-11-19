"""Repeat the matched batching experiment using locally available MiniLM weights."""
import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.nn.functional import cross_entropy

from .common import ROOT, print_json, read_json
from .relation_binding import decode, score
from .relation_model import Features, RelationHead


def validate_schedule(schedule, window):
    a, b = schedule["grouped"], schedule["mixed"]
    if len(a) != len(b) or not a or len(a) % window or any(len(batch) != 8 for batch in a + b):
        raise ValueError("Schedules must use equal numbers of eight-document updates.")
    for start in range(0, len(a), window):
        left = Counter(i for batch in a[start:start+window] for i in batch)
        right = Counter(i for batch in b[start:start+window] for i in batch)
        if left != right:
            raise ValueError(f"Exposure differs in the window starting at update {start}.")


def train(features, labels, schedule, seed, learning_rate, device):
    torch.manual_seed(seed)
    model = RelationHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for batch in schedule:
        logits, _ = model(features[batch].flatten(0, 1))
        loss = cross_entropy(logits, labels[batch].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def evaluate(model, features, rows):
    scores, _ = model(features)
    predictions = []
    for row, values in zip(rows, scores.cpu().numpy()):
        predictions.append({"id": row["id"], "edges": decode(row["candidates"], values),
                            "pair_classes": values.argmax(axis=1).reshape(2, 3).tolist()})
    return score(rows, predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, choices=(32, 43), default=32)
    parser.add_argument("--smoke", action="store_true", help="One matched window and 32 test documents.")
    args = parser.parse_args()
    config = read_json(ROOT / "configs" / "relations.json")
    data = ROOT / "data" / "relations"
    train_rows = read_json(data / "train.json.gz")
    test_rows = read_json(data / "test.json.gz")
    schedule = read_json(data / f"schedule_{args.seed}.json")
    window = config["window_updates"]
    validate_schedule(schedule, window)
    if len(schedule["grouped"]) != config["updates"]:
        raise ValueError("Archived schedule does not match the experiment configuration.")
    if args.smoke:
        # Balance presentation counts over the whole window, not just the first step.
        schedule = {name: batches[:window] for name, batches in schedule.items()}
        used = sorted({i for batch in schedule["grouped"] for i in batch})
        remap = {old: new for new, old in enumerate(used)}
        schedule = {name: [[remap[i] for i in batch] for batch in batches]
                    for name, batches in schedule.items()}
        train_rows = [train_rows[i] for i in used]
        test_rows = test_rows[:32]
    encoder = Features(str(args.model_dir.resolve()), args.device)
    batch_size = config["encoder_batch_size"]
    train_x = encoder.encode(train_rows, batch_size)
    test_x = encoder.encode(test_rows, batch_size)
    labels = torch.tensor([row["classes"] for row in train_rows], device=args.device).reshape(-1, 6)
    result = {"seed": args.seed, "device": args.device, "smoke": args.smoke,
              "updates_per_arm": len(schedule["grouped"]), "methods": {}}
    for name, batches in schedule.items():
        model = train(train_x, labels, batches, args.seed, config["learning_rate"], args.device)
        result["methods"][name] = evaluate(model, test_x, test_rows)
    print_json(result)


if __name__ == "__main__":
    main()
