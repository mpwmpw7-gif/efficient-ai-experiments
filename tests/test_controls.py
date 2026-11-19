"""Small controls for scoring, evidence updates and feature construction."""
import unittest

import numpy as np

from experiments.common import grouped_difference
from experiments.decision_first import parse_state
from experiments.relation_binding import decode, execute
from experiments.visual_evidence import posterior_history
from experiments.visual_features import aligned, unaligned


class ControlsTest(unittest.TestCase):
    def test_generation_parser_requires_complete_state_and_rejects_contradiction(self):
        self.assertEqual(parse_state("The book is blue and in the library."),
                         {"object": "book", "color": "blue", "location": "library"})
        self.assertIsNone(parse_state("The book is in the library."))
        self.assertIsNone(parse_state("The book is not blue and is in the library."))
        self.assertEqual(parse_state("There is not enough information."), "unknown")

    def test_relation_decode_uses_forward_margin_and_polarity(self):
        evidence = np.zeros((6, 5))
        evidence[1, 2] = 3
        evidence[1, 4] = 1
        evidence[3, 1] = 2
        evidence[3, 3] = 4
        edges = decode(["Aster", "Briar", "Cedar"], evidence)
        self.assertEqual(edges, [["Briar", True], None])
        spans = [{"text": "Order Briar is cancelled."},
                 {"text": "The cancellation of order Briar has been revoked."}]
        self.assertEqual(execute(spans, edges), [False, None])

    def test_grouped_estimate_weights_groups_equally(self):
        result = grouped_difference([1, 1, 1, 0], [0, 0, 0, 0], ["A", "A", "A", "B"])
        self.assertEqual(result["difference"], 0.5)

    def test_rejected_and_zero_evidence_do_not_change_an_established_belief(self):
        mask = np.ones((1, 4), dtype=bool)
        probabilities = posterior_history(
            np.array([3.0, 0.0, 0.05, -4.0]), mask, deadband=0.1, switch_prior=0.1,
        )[0]
        self.assertGreater(probabilities[1], 0.9)
        self.assertEqual(probabilities[1], probabilities[2])
        self.assertEqual(probabilities[2], probabilities[3])
        self.assertLess(probabilities[4], 0.5)

    def test_removing_alignment_preserves_intensity_histograms_within_cells(self):
        before = np.zeros((1, 56, 56, 3), dtype=np.uint8)
        before[0, 24, 12] = 255
        after = before.copy()
        after[0, 24, 12] = 0
        after[0, 24, 13] = 255
        a = aligned(before, np.array([2]), after)
        b = unaligned(before, np.array([2]), after)
        self.assertEqual(a.shape, (1, 56))
        np.testing.assert_array_equal(a[:, :40], b[:, :40])
        self.assertGreater(a[0, 40], 0)
        self.assertEqual(b[0, 40], 0)
        np.testing.assert_array_equal(aligned(before, np.array([2]), before),
                                      unaligned(before, np.array([2]), before))


if __name__ == "__main__":
    unittest.main()
