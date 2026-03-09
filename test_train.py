import sys
from unittest.mock import MagicMock

# Mock out kernels and torch.cuda to avoid requiring an NVIDIA GPU for tests
sys.modules['kernels'] = MagicMock()
import torch
torch.cuda.get_device_capability = MagicMock(return_value=(8, 0))
torch.cuda.is_available = MagicMock(return_value=False)

import pytest
import train

def test_get_weight_decay():
    """
    Test the get_weight_decay function from train.py.
    The function signature is: def get_weight_decay(progress): return WEIGHT_DECAY * (1 - progress)
    """
    # Using the current WEIGHT_DECAY from train.py
    wd = train.WEIGHT_DECAY

    # Test start of training (progress = 0.0)
    assert train.get_weight_decay(0.0) == wd * 1.0

    # Test mid training (progress = 0.5)
    assert train.get_weight_decay(0.5) == wd * 0.5

    # Test end of training (progress = 1.0)
    assert train.get_weight_decay(1.0) == 0.0

    # Test an intermediate value (progress = 0.25)
    assert train.get_weight_decay(0.25) == wd * 0.75

    # Test out of bounds - though progress usually clamped [0, 1] in training loop
    assert train.get_weight_decay(1.5) == wd * -0.5
    assert train.get_weight_decay(-0.5) == wd * 1.5

if __name__ == "__main__":
    pytest.main([__file__])
