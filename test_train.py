import pytest
import unittest.mock
import sys

# Mock kernels to avoid needing it
sys.modules['kernels'] = unittest.mock.MagicMock()

import torch
# Patch torch.cuda.get_device_capability to avoid needing a GPU
torch.cuda.get_device_capability = unittest.mock.MagicMock(return_value=(8, 0))

from train import get_muon_momentum

def test_get_muon_momentum():
    assert get_muon_momentum(0) == 0.85
    assert abs(get_muon_momentum(150) - 0.90) < 1e-5
    assert get_muon_momentum(300) == 0.95
    assert get_muon_momentum(600) == 0.95
    assert get_muon_momentum(3000) == 0.95

    # Negative steps aren't expected in training, but mathematically:
    # frac = min(-100/300, 1) = -0.3333333333333333
    # (1 - -0.3333)*0.85 + (-0.3333)*0.95 = 1.3333*0.85 - 0.3333*0.95
    # = 1.133333333 - 0.3166666666 = 0.8166666666666667
    assert abs(get_muon_momentum(-100) - 0.8166666666666667) < 1e-5
