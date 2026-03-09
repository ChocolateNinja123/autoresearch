import sys
from unittest.mock import MagicMock

# Mock torch and other dependencies before importing train
mock_torch = MagicMock()
mock_f = MagicMock()
mock_torch.nn.functional = mock_f
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.nn.functional"] = mock_f
sys.modules["kernels"] = MagicMock()
sys.modules["prepare"] = MagicMock()

import train

def test_has_ve():
    print("Testing has_ve...")
    # For n_layer = 12 (even), (n_layer-1)%2 = 1
    assert train.has_ve(0, 12) == False
    assert train.has_ve(1, 12) == True
    assert train.has_ve(10, 12) == False
    assert train.has_ve(11, 12) == True # Last layer

    # For n_layer = 13 (odd), (n_layer-1)%2 = 0
    assert train.has_ve(0, 13) == True
    assert train.has_ve(1, 13) == False
    assert train.has_ve(11, 13) == False
    assert train.has_ve(12, 13) == True # Last layer
    print("has_ve tests passed!")

def test_norm():
    print("Testing norm...")
    x = MagicMock()
    x.size.return_value = 768

    train.norm(x)

    # Verify F.rms_norm was called with x and (x.size(-1),)
    mock_f.rms_norm.assert_called_once()
    args, kwargs = mock_f.rms_norm.call_args
    assert args[0] == x
    assert args[1] == (768,)
    print("norm tests passed!")

if __name__ == "__main__":
    try:
        test_has_ve()
        test_norm()
        print("All tests passed successfully!")
    except Exception as e:
        print(f"Tests failed: {e}")
        sys.exit(1)
