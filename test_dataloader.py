import prepare
import torch

# Patch torch.empty to avoid cuda
original_empty = torch.empty
def mocked_empty(*args, **kwargs):
    kwargs.pop("device", None)
    kwargs.pop("pin_memory", None)
    return original_empty(*args, **kwargs)

torch.empty = mocked_empty

tokenizer = prepare.Tokenizer.from_directory(prepare.TOKENIZER_DIR)
dataloader = prepare.make_dataloader(tokenizer, B=4, T=128, split="train")

x, y, epoch = next(dataloader)

print("x shape:", x.shape)
print("y shape:", y.shape)
print("epoch:", epoch)
print("dataloader output x sum:", x.sum().item())
print("dataloader output y sum:", y.sum().item())
