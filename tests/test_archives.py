"""Guard the reported portfolio figures against archived predictions and fresh fits."""
import unittest

from experiments.common import ROOT, read_json
from experiments.decision_first import assess
from experiments.relation_binding import audit_archive
from experiments.visual_evidence import run


# A fixed-budget L-BFGS refit can take slightly different steps with different
# BLAS kernels. The largest observed replay drift was 0.00211 bits/outcome.
REFIT_BITS_TOLERANCE = 0.005


class ArchivedResultsTest(unittest.TestCase):
    def test_decision_first_replays_archived_text(self):
        fixture = read_json(ROOT / "data" / "decision_outputs.json")
        baseline = read_json(ROOT / "results" / "decision_first.json")
        for study, fields in [("state_generation", False), ("plan_serialization", True)]:
            with self.subTest(study=study):
                replay = assess(fixture[study], fields=fields)["methods"]
                for method, row in replay.items():
                    expected = baseline[study][method]
                    self.assertEqual(row["cases"], expected["n"])
                    self.assertAlmostEqual(row["correct"] / row["cases"], expected["accuracy"])
                    self.assertAlmostEqual(row["complete"] / row["cases"], expected["complete"])

    def test_relation_replays_signed_claims_and_executes_edges(self):
        actual = audit_archive(ROOT / "data" / "relations", (32, 43))
        expected = read_json(ROOT / "results" / "relation_binding.json")
        for seed, arms in actual.items():
            for name, scores in arms.items():
                with self.subTest(seed=seed, arm=name):
                    original = expected["seeds"][seed][name]["splits"]["confirmation"]
                    self.assertEqual(scores["oracle_states_correct"], original["docs"])
                    self.assertEqual(scores["forward_edges_correct"], original["forward"])
                    self.assertEqual(scores["signed_reverse_correct"], original["reverse"])
                    self.assertEqual(scores["signed_reverse_total"], original["reverse_n"])
                    self.assertEqual(scores["states_correct"], original["states"])
                    self.assertEqual(scores["correct_states_with_incorrect_edges"], original["compensating_states"])
                    self.assertEqual(scores["complete_swaps_correct"],
                                     original["paired"]["governing_swap"]["both_edge_sets_correct"])

    def test_visual_model_refits_all_splits(self):
        replay = run(ROOT / "data" / "visual", read_json(ROOT / "configs" / "visual.json"))
        historical = read_json(ROOT / "results" / "visual_evidence.json")
        for name, split in replay["splits"].items():
            old = historical["families"][name]
            self.assertEqual(split["layouts"], old["groups"])
            for arm, key in [("original", "A0"), ("aligned", "A1"), ("unaligned", "A2")]:
                for calibration in ("raw", "calibrated"):
                    with self.subTest(split=name, arm=arm, calibration=calibration):
                        actual = split["methods"][arm][calibration]
                        expected = old["methods"][key][calibration]
                        self.assertAlmostEqual(actual["bits_per_outcome"],
                                               expected["outcome"]["bits"], delta=REFIT_BITS_TOLERANCE)
                        # The archived multi-class Brier convention sums two components.
                        self.assertAlmostEqual(2 * actual["binary_brier"], expected["outcome"]["brier"], delta=0.001)
                        self.assertEqual(actual["neutral_at_least_3_to_1"],
                                         expected["events"]["neutral_false_3to1"])
                        self.assertEqual(actual["identifiable_direction_correct"],
                                         round(expected["events"]["identifiable_events"]
                                               * expected["events"]["identifiable_dir_accuracy"]))
            self.assertAlmostEqual(split["methods"]["geometry"]["bits_per_outcome"], old["geometry"]["bits"], places=5)


if __name__ == "__main__":
    unittest.main()
