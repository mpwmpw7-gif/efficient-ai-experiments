"""Training-path checks, enabled when the optional model stack is installed."""
import importlib.util
import unittest

from experiments.common import ROOT, read_json


with_torch = all(importlib.util.find_spec(pkg) for pkg in ("torch", "transformers"))


@unittest.skipUnless(with_torch, "install requirements-relation.txt for PyTorch tests")
class TrainingTest(unittest.TestCase):
    def test_readout_shape_and_gradient(self):
        import torch
        from experiments.relation_model import RelationHead

        head = RelationHead()
        x = torch.zeros(2, 6, 2304)
        score, embedding = head(x)
        self.assertEqual(score.shape, (2, 6, 5))
        self.assertEqual(embedding.shape, (2, 6, 32))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 224549)
        loss = torch.nn.functional.cross_entropy(score.flatten(0, 1), torch.zeros(12, dtype=torch.long))
        loss.backward()
        self.assertGreater(head.edge.weight.grad.abs().sum().item(), 0)

    def test_both_historical_schedules_match_exposure(self):
        from experiments.train_relations import validate_schedule

        root = ROOT / "data" / "relations"
        for seed in (32, 43):
            schedule = read_json(root / f"schedule_{seed}.json")
            validate_schedule(schedule, window=24)
            self.assertEqual(len(schedule["grouped"]), 360)


if __name__ == "__main__":
    unittest.main()
