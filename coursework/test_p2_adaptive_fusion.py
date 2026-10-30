"""Tests for the learnable P2 branch fusion module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    from ultralytics.nn.modules import AdaptiveWeightedConcat

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class AdaptiveWeightedConcatTests(unittest.TestCase):
    def test_initial_output_matches_concat(self):
        features = [
            torch.randn(2, 3, 8, 8),
            torch.randn(2, 5, 8, 8),
            torch.randn(2, 7, 8, 8),
        ]
        module = AdaptiveWeightedConcat(n_inputs=3)

        actual = module(features)
        expected = torch.cat(features, dim=1)

        self.assertTrue(torch.equal(actual, expected))

    def test_branch_weights_receive_gradients(self):
        features = [torch.randn(1, 2, 4, 4) for _ in range(3)]
        module = AdaptiveWeightedConcat(n_inputs=3)

        module(features).square().mean().backward()

        self.assertIsNotNone(module.logits.grad)
        self.assertTrue(torch.isfinite(module.logits.grad).all())

    def test_rejects_wrong_input_count(self):
        module = AdaptiveWeightedConcat(n_inputs=3)
        with self.assertRaises(ValueError):
            module([torch.randn(1, 2, 4, 4)])


if __name__ == "__main__":
    unittest.main()
