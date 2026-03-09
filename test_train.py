import unittest
from unittest.mock import patch

# Mock out CUDA related stuff before importing train
import torch
import sys

# Patch torch.empty to avoid cuda
original_empty = torch.empty
def mocked_empty(*args, **kwargs):
    kwargs.pop("device", None)
    kwargs.pop("pin_memory", None)
    return original_empty(*args, **kwargs)

# Safely import train by mocking torch features only during the import
with patch('torch.empty', side_effect=mocked_empty):
    with patch('torch.cuda.get_device_capability', return_value=(8, 0)):
        import train

class TestGetLrMultiplier(unittest.TestCase):
    def test_with_warmup_and_warmdown(self):
        # Patch the constants for a typical scenario with both warmup and warmdown
        with patch('train.WARMUP_RATIO', 0.1), \
             patch('train.WARMDOWN_RATIO', 0.2), \
             patch('train.FINAL_LR_FRAC', 0.1):

            # Progress < WARMUP_RATIO (0.0 to 0.1)
            self.assertAlmostEqual(train.get_lr_multiplier(0.0), 0.0)
            self.assertAlmostEqual(train.get_lr_multiplier(0.05), 0.5)
            self.assertAlmostEqual(train.get_lr_multiplier(0.099), 0.99)

            # Progress between WARMUP_RATIO and (1.0 - WARMDOWN_RATIO) (0.1 to 0.8)
            self.assertAlmostEqual(train.get_lr_multiplier(0.1), 1.0)
            self.assertAlmostEqual(train.get_lr_multiplier(0.5), 1.0)
            self.assertAlmostEqual(train.get_lr_multiplier(0.799), 1.0)

            # Progress >= (1.0 - WARMDOWN_RATIO) (0.8 to 1.0)
            # Formula: cooldown = (1.0 - progress) / WARMDOWN_RATIO
            # LR = cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

            # At 0.8: cooldown = 0.2 / 0.2 = 1.0 -> LR = 1.0 * 1.0 + 0.0 = 1.0
            self.assertAlmostEqual(train.get_lr_multiplier(0.8), 1.0)

            # At 0.9: cooldown = 0.1 / 0.2 = 0.5 -> LR = 0.5 * 1.0 + 0.5 * 0.1 = 0.55
            self.assertAlmostEqual(train.get_lr_multiplier(0.9), 0.55)

            # At 1.0: cooldown = 0.0 / 0.2 = 0.0 -> LR = 0.0 * 1.0 + 1.0 * 0.1 = 0.1
            self.assertAlmostEqual(train.get_lr_multiplier(1.0), 0.1)

    def test_no_warmup(self):
        # Patch the constants for a scenario with no warmup (like current default)
        with patch('train.WARMUP_RATIO', 0.0), \
             patch('train.WARMDOWN_RATIO', 0.5), \
             patch('train.FINAL_LR_FRAC', 0.0):

            # Progress starts at 1.0 immediately because WARMUP_RATIO is 0
            # and progress (0.0) is not < 0.0
            self.assertAlmostEqual(train.get_lr_multiplier(0.0), 1.0)
            self.assertAlmostEqual(train.get_lr_multiplier(0.25), 1.0)
            self.assertAlmostEqual(train.get_lr_multiplier(0.499), 1.0)

            # Progress >= 0.5 (warmdown)
            self.assertAlmostEqual(train.get_lr_multiplier(0.5), 1.0)
            self.assertAlmostEqual(train.get_lr_multiplier(0.75), 0.5)
            self.assertAlmostEqual(train.get_lr_multiplier(1.0), 0.0)

    def test_no_warmdown(self):
        # Scenario with warmup but no warmdown
        with patch('train.WARMUP_RATIO', 0.2), \
             patch('train.WARMDOWN_RATIO', 0.0), \
             patch('train.FINAL_LR_FRAC', 0.1):

            # Progress < WARMUP_RATIO
            self.assertAlmostEqual(train.get_lr_multiplier(0.0), 0.0)
            self.assertAlmostEqual(train.get_lr_multiplier(0.1), 0.5)

            # Progress >= WARMUP_RATIO
            # If WARMDOWN_RATIO is 0.0, 1.0 - WARMDOWN_RATIO = 1.0
            # So progress < 1.0 will return 1.0
            self.assertAlmostEqual(train.get_lr_multiplier(0.2), 1.0)
            self.assertAlmostEqual(train.get_lr_multiplier(0.99), 1.0)

            # If progress is exactly 1.0 (which shouldn't happen during loop due to `progress < 1.0` in train, but just in case)
            # wait, if progress is 1.0, and WARMDOWN_RATIO is 0.0
            # progress < 1.0 is False
            # it goes to else: cooldown = (1.0 - 1.0) / 0.0 -> ZeroDivisionError
            # Since the function isn't protected against this edge case mathematically without checking WARMDOWN_RATIO,
            # we will test up to 0.999999 as standard progress behavior.
            # We won't test exactly 1.0 here unless the codebase expects it to not crash.
            # Actually, let's just assert that it raises ZeroDivisionError or test only < 1.0
            with self.assertRaises(ZeroDivisionError):
                train.get_lr_multiplier(1.0)

    def test_no_warmup_no_warmdown(self):
        # Scenario where learning rate is constant throughout the training
        with patch('train.WARMUP_RATIO', 0.0), \
             patch('train.WARMDOWN_RATIO', 0.0), \
             patch('train.FINAL_LR_FRAC', 0.1):

            self.assertAlmostEqual(train.get_lr_multiplier(0.0), 1.0)
            self.assertAlmostEqual(train.get_lr_multiplier(0.5), 1.0)
            self.assertAlmostEqual(train.get_lr_multiplier(0.99), 1.0)

    def test_edge_cases(self):
        # Test progress values slightly outside bounds
        with patch('train.WARMUP_RATIO', 0.1), \
             patch('train.WARMDOWN_RATIO', 0.2), \
             patch('train.FINAL_LR_FRAC', 0.1):

            # Progress < 0.0 (Should never happen, but test logic anyway: returns < 0)
            self.assertAlmostEqual(train.get_lr_multiplier(-0.1), -1.0)

            # Progress > 1.0 (Should not normally happen, but let's test extrapolation)
            # cooldown = (1.0 - 1.1) / 0.2 = -0.5
            # LR = (-0.5) * 1.0 + (1.5) * 0.1 = -0.5 + 0.15 = -0.35
            self.assertAlmostEqual(train.get_lr_multiplier(1.1), -0.35)

if __name__ == '__main__':
    unittest.main()
