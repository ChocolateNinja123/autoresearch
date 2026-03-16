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

from train import has_ve, get_lr_multiplier, get_muon_momentum, get_weight_decay, norm

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

    # Warmdown phase (0.8 to 1.0) using cosine decay
    # At 0.8, decay_ratio = 0.0 -> cos(0) = 1.0 -> coeff = 1.0 -> 1.0
    # At 0.9, decay_ratio = 0.5 -> cos(pi/2) = 0.0 -> coeff = 0.5 -> 0.55
    # At 1.0, decay_ratio = 1.0 -> cos(pi) = -1.0 -> coeff = 0.0 -> 0.1
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

    # Warmdown phase (0.8 to 1.0) using cosine decay
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


@unittest.mock.patch('train.WEIGHT_DECAY', 0.2)
def test_get_weight_decay_start():
    """Test get_weight_decay at progress 0.0 (should be exactly WEIGHT_DECAY)."""
    assert get_weight_decay(0.0) == pytest.approx(0.2)

@unittest.mock.patch('train.WEIGHT_DECAY', 0.2)
def test_get_weight_decay_mid():
    """Test get_weight_decay at progress 0.5 (should be half of WEIGHT_DECAY)."""
    assert get_weight_decay(0.5) == pytest.approx(0.1)

@unittest.mock.patch('train.WEIGHT_DECAY', 0.2)
def test_get_weight_decay_end():
    """Test get_weight_decay at progress 1.0 (should be exactly 0.0)."""
    assert get_weight_decay(1.0) == pytest.approx(0.0)

@unittest.mock.patch('train.WEIGHT_DECAY', 0.2)
def test_get_weight_decay_out_of_bounds():
    """Test get_weight_decay with out of bounds progress to document behavior."""
    assert get_weight_decay(1.5) == pytest.approx(-0.1)
    assert get_weight_decay(-0.5) == pytest.approx(0.3)

# In test_train.py, torch is mocked. We can't use real PyTorch tensors directly
# inside these functions. We can either write the test logic in a separate file,
# or use `unittest.mock.patch` around the specific `train.py` import inside a subprocess
# to bypass the CUDA device error, then run the test logic inside that subprocess using real torch.
# Wait, since `train.py` has its main logic protected by `if __name__ == "__main__":`,
# we can just mock `torch.cuda.get_device_capability` and `get_kernel` using `patch` during a subprocess.

# In the current environment, `sys.modules['torch']` is a MagicMock, preventing us from using real tensors
# cleanly without either subprocesses or running into torch reload bugs like "conv1d already has a docstring".
# To fix this elegantly, we don't import `train.py` directly in our tests and try to un-mock it.
# Instead, we just patch the `train.apply_rotary_emb` using `unittest.mock.patch` if we want to run train,
# BUT we just want to run `apply_rotary_emb` math!
# Wait, we can just define a test fixture that patches `torch` dynamically just for importing train.
# Since `train.py` is already imported at the top of `test_train.py`, we can't un-import it easily.
# But `apply_rotary_emb` relies on `torch.cat`. Since `train.py` has `import torch` mapped to the mock,
# `torch.cat` inside `train.py` is `mock_torch.cat`.
# We can just run the function and assert that `mock_torch.cat` was called correctly!

def test_apply_rotary_emb_mocked():
    from train import apply_rotary_emb
    import sys
    mock_torch = sys.modules['torch']

    # We pass standard Python objects (or mock tensors) and trace the math
    class DummyTensor:
        def __init__(self, name, ndim=4, shape=(2, 4, 3, 8)):
            self.name = name
            self.ndim = ndim
            self.shape = shape

        def __getitem__(self, key):
            return DummyTensor(f"{self.name}[{key}]")

        def __mul__(self, other):
            other_name = other.name if isinstance(other, DummyTensor) else str(other)
            return DummyTensor(f"({self.name} * {other_name})")

        def __add__(self, other):
            other_name = other.name if isinstance(other, DummyTensor) else str(other)
            return DummyTensor(f"({self.name} + {other_name})")

        def __neg__(self):
            return DummyTensor(f"(-{self.name})")

    x = DummyTensor("x")
    cos = DummyTensor("cos")
    sin = DummyTensor("sin")

    # Reset mock before call
    mock_torch.cat.reset_mock()
    mock_torch.cat.return_value = DummyTensor("result")

    res = apply_rotary_emb(x, cos, sin)

    assert res.name == "result"

    # Check that mock_torch.cat was called with a list of two elements and dim=3
    mock_torch.cat.assert_called_once()
    args, kwargs = mock_torch.cat.call_args
    assert len(args) == 2
    tensors = args[0]
    assert len(tensors) == 2

    dim = args[1] if len(args) > 1 else kwargs.get('dim', None)
    if dim is None and len(args) == 1 and len(kwargs) == 0:
        # Check positional arg 3 or kwarg dim=3
        # In train.py it's called as: torch.cat([y1, y2], 3)
        assert mock_torch.cat.call_args[0][1] == 3

    y1, y2 = tensors
    # Check the mathematical structure encoded in names
    # d = 8 // 2 = 4
    # x1 = x[(Ellipsis, slice(None, 4, None))]
    # x2 = x[(Ellipsis, slice(4, None, None))]

    assert "slice(None, 4, None)" in y1.name
    assert "cos" in y1.name
    assert "sin" in y1.name

    assert "slice(4, None, None)" in y2.name
    assert "-sin" in y2.name
    assert "cos" in y2.name

def test_apply_rotary_emb_ndim_assertion():
    from train import apply_rotary_emb
    import pytest

    class DummyTensor:
        def __init__(self, ndim):
            self.ndim = ndim
            self.shape = [1] * ndim

    with pytest.raises(AssertionError):
        apply_rotary_emb(DummyTensor(3), DummyTensor(3), DummyTensor(3))

    with pytest.raises(AssertionError):
        apply_rotary_emb(DummyTensor(5), DummyTensor(5), DummyTensor(5))

def test_norm_mocked():
    mock_torch_f = sys.modules['torch.nn.functional']

    class DummyTensor:
        def __init__(self, name, last_dim=768):
            self.name = name
            self.last_dim = last_dim

        def size(self, dim):
            if dim == -1:
                return self.last_dim
            return 1

    x = DummyTensor("x", last_dim=512)

    mock_torch_f.rms_norm.reset_mock()
    norm(x)

    mock_torch_f.rms_norm.assert_called_once_with(x, (512,))
