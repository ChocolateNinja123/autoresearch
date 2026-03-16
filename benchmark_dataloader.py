import time
import prepare
import torch
import unittest.mock

# Patch torch.empty to avoid cuda and pin_memory issues when testing without GPU
original_empty = torch.empty
def mocked_empty(*args, **kwargs):
    kwargs.pop("device", None)
    kwargs.pop("pin_memory", None)
    return original_empty(*args, **kwargs)

torch.empty = mocked_empty

class MockTokenizer:
    def get_bos_token_id(self):
        return 0

# Mock the parquet reading part to just yield fast dummy data
def mock_document_batches(split, tokenizer_batch_size=128):
    epoch = 1
    # Create a dummy batch of documents of varying lengths
    batch = ["a" * i for i in range(1, 100)] * 5
    while True:
        for i in range(0, len(batch), tokenizer_batch_size):
            yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1

prepare._document_batches = mock_document_batches

def run_benchmark():
    tokenizer = MockTokenizer()

    # Large batch size and seq len to stress the dataloader packing logic
    B = 32
    T = 2048

    dataloader = prepare.make_dataloader(tokenizer, B, T, split="train", buffer_size=5000)

    # Warmup
    for _ in range(5):
        next(dataloader)

    start_time = time.time()
    num_batches = 50

    for _ in range(num_batches):
        next(dataloader)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Time to generate {num_batches} batches: {duration:.4f} seconds")
    print(f"Batches per second: {num_batches / duration:.2f}")

if __name__ == "__main__":
    run_benchmark()
