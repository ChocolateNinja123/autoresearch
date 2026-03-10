import prepare
import torch

# Patch torch.empty to avoid cuda
original_empty = torch.empty
def mocked_empty(*args, **kwargs):
    kwargs.pop("device", None)
    kwargs.pop("pin_memory", None)
    return original_empty(*args, **kwargs)

torch.empty = mocked_empty

def test_dataloader_basic():
    tokenizer = prepare.Tokenizer.from_directory(prepare.TOKENIZER_DIR)
    dataloader = prepare.make_dataloader(tokenizer, B=4, T=128, split="train")

    x, y, epoch = next(dataloader)

    assert x.shape == (4, 128)
    assert y.shape == (4, 128)
    assert epoch == 1
    assert x.sum().item() > 0
    assert y.sum().item() > 0
