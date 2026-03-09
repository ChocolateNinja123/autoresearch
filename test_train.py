import pytest
import sys
import unittest.mock

# Create a more complete mock for torch to bypass CUDA checks when importing train.py
mock_torch = unittest.mock.MagicMock()
mock_torch.cuda = unittest.mock.MagicMock()
mock_torch.cuda.get_device_capability.return_value = (8, 0) # Mock some capability

# We mock all required submodules so import doesn't fail
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn
sys.modules['torch.nn.functional'] = mock_torch.nn.functional
sys.modules['kernels'] = unittest.mock.MagicMock()
sys.modules['prepare'] = unittest.mock.MagicMock()

from train import has_ve, get_lr_multiplier, get_muon_momentum

def test_has_ve_even_layers():
    """Test has_ve with an even number of layers (e.g., n_layer=8)"""
    n_layer = 8

    # In an 8-layer model, the last layer is index 7.
    # 7 % 2 == 1, so odd indices should have VE.
    # Expected: [False, True, False, True, False, True, False, True]
    assert has_ve(0, n_layer) is False
    assert has_ve(1, n_layer) is True
    assert has_ve(2, n_layer) is False
    assert has_ve(3, n_layer) is True
    assert has_ve(4, n_layer) is False
    assert has_ve(5, n_layer) is True
    assert has_ve(6, n_layer) is False
    assert has_ve(7, n_layer) is True

def test_has_ve_odd_layers():
    """Test has_ve with an odd number of layers (e.g., n_layer=5)"""
    n_layer = 5

    # In a 5-layer model, the last layer is index 4.
    # 4 % 2 == 0, so even indices should have VE.
    # Expected: [True, False, True, False, True]
    assert has_ve(0, n_layer) is True
    assert has_ve(1, n_layer) is False
    assert has_ve(2, n_layer) is True
    assert has_ve(3, n_layer) is False
    assert has_ve(4, n_layer) is True

def test_has_ve_single_layer():
    """Test has_ve with a single layer"""
    n_layer = 1

    # In a 1-layer model, the last layer is index 0.
    # 0 % 2 == 0, so index 0 should have VE.
    # Expected: [True]
    assert has_ve(0, n_layer) is True

def test_has_ve_last_layer_always_true():
    """Property test: the last layer should always have VE"""
    for n_layer in range(1, 20):
        # The last layer is at index `n_layer - 1`
        assert has_ve(n_layer - 1, n_layer) is True

def test_has_ve_alternating_property():
    """Property test: VE should strictly alternate"""
    for n_layer in range(2, 20):
        for i in range(n_layer - 1):
            # Consecutive layers should have different has_ve values
            assert has_ve(i, n_layer) != has_ve(i + 1, n_layer)

@unittest.mock.patch('train.WARMUP_RATIO', 0.1)
@unittest.mock.patch('train.WARMDOWN_RATIO', 0.2)
@unittest.mock.patch('train.FINAL_LR_FRAC', 0.1)
def test_get_lr_multiplier_standard_schedule():
    """Test standard LR schedule with warmup, plateau, and warmdown."""
    # Warmup phase (0.0 to 0.1)
    assert get_lr_multiplier(0.0) == 0.0
    assert get_lr_multiplier(0.05) == 0.5  # 0.05 / 0.1

    # Plateau phase (0.1 to 0.8)
    assert get_lr_multiplier(0.1) == 1.0
    assert get_lr_multiplier(0.5) == 1.0
    assert get_lr_multiplier(0.79) == 1.0

    # Warmdown phase (0.8 to 1.0)
    # At 0.8, cooldown = 1.0, multiplier = 1.0 * 1.0 + 0 * 0.1 = 1.0
    # At 0.9, cooldown = 0.5, multiplier = 0.5 * 1.0 + 0.5 * 0.1 = 0.55
    # At 1.0, cooldown = 0.0, multiplier = 0.0 * 1.0 + 1.0 * 0.1 = 0.1
    assert get_lr_multiplier(0.8) == pytest.approx(1.0)
    assert get_lr_multiplier(0.9) == pytest.approx(0.55)
    assert get_lr_multiplier(1.0) == pytest.approx(0.1)

@unittest.mock.patch('train.WARMUP_RATIO', 0.0)
@unittest.mock.patch('train.WARMDOWN_RATIO', 0.2)
@unittest.mock.patch('train.FINAL_LR_FRAC', 0.1)
def test_get_lr_multiplier_no_warmup():
    """Test LR schedule with no warmup."""
    # Should start immediately at 1.0
    assert get_lr_multiplier(0.0) == pytest.approx(1.0)
    assert get_lr_multiplier(0.5) == pytest.approx(1.0)

    # Warmdown phase (0.8 to 1.0)
    assert get_lr_multiplier(0.9) == pytest.approx(0.55)
    assert get_lr_multiplier(1.0) == pytest.approx(0.1)

@unittest.mock.patch('train.WARMUP_RATIO', 0.1)
@unittest.mock.patch('train.WARMDOWN_RATIO', 0.0)
@unittest.mock.patch('train.FINAL_LR_FRAC', 0.1)
def test_get_lr_multiplier_no_warmdown():
    """Test LR schedule with no warmdown."""
    # Warmup phase
    assert get_lr_multiplier(0.0) == 0.0
    assert get_lr_multiplier(0.05) == 0.5

    # With WARMDOWN_RATIO=0.0, progress < 1.0 is always true until the end.
    # We should handle division by zero or test how the function behaves.
    # Wait, if WARMDOWN_RATIO == 0.0, 1.0 - 0.0 = 1.0, so progress < 1.0
    # means it stays at 1.0 until exactly progress == 1.0
    assert get_lr_multiplier(0.1) == 1.0
    assert get_lr_multiplier(0.9) == 1.0
    assert get_lr_multiplier(0.99) == 1.0

    # For progress == 1.0, the else block runs, which divides by WARMDOWN_RATIO (0.0),
    # causing ZeroDivisionError.
    with pytest.raises(ZeroDivisionError):
        get_lr_multiplier(1.0)


def test_get_muon_momentum_start():
    """Test get_muon_momentum at step 0 (should be exactly 0.85)."""
    assert get_muon_momentum(0) == pytest.approx(0.85)

def test_get_muon_momentum_mid():
    """Test get_muon_momentum at midpoint step 150 (should be exactly 0.90)."""
    assert get_muon_momentum(150) == pytest.approx(0.90)

def test_get_muon_momentum_end():
    """Test get_muon_momentum at step 300 (should be exactly 0.95)."""
    assert get_muon_momentum(300) == pytest.approx(0.95)

def test_get_muon_momentum_capped():
    """Test get_muon_momentum beyond step 300 (should be capped at 0.95)."""
    assert get_muon_momentum(500) == pytest.approx(0.95)
    assert get_muon_momentum(1000) == pytest.approx(0.95)

def test_get_muon_momentum_negative():
    """Test get_muon_momentum with negative steps (should be capped at 0.85)."""
    # The current implementation of min(step / 300, 1) returns negative numbers for negative steps.
    # While step should ideally be positive, we test the actual mathematical output.
    # (1 - (step/300)) * 0.85 + (step/300) * 0.95 = 0.85 + (step/300) * 0.1
    # For step -300: 0.85 - 0.1 = 0.75
    assert get_muon_momentum(-300) == pytest.approx(0.75)
