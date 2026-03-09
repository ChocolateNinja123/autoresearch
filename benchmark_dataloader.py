import time
import torch
import prepare

# Patch torch.empty to avoid cuda
original_empty = torch.empty
def mocked_empty(*args, **kwargs):
    kwargs.pop("device", None)
    kwargs.pop("pin_memory", None)
    return original_empty(*args, **kwargs)

torch.empty = mocked_empty

def run_benchmark():
    tokenizer = prepare.Tokenizer.from_directory(prepare.TOKENIZER_DIR)

    # Use huge buffer
    dataloader = prepare.make_dataloader(tokenizer, B=128, T=2048, split="train", buffer_size=10000)

    # Warmup
    next(dataloader)

    start = time.time()
    for i in range(10):
        try:
            next(dataloader)
        except StopIteration:
            break
    end = time.time()
    print(f"Time: {end - start:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
