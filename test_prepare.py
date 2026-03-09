import unittest
import os
import json
import tempfile
from unittest.mock import patch
import torch

import prepare

class TestGetTokenBytes(unittest.TestCase):
    def setUp(self):
        # Patch torch.empty to avoid cuda errors locally if needed, similar to test_dataloader.py
        self.original_empty = torch.empty
        def mocked_empty(*args, **kwargs):
            kwargs.pop("device", None)
            kwargs.pop("pin_memory", None)
            return self.original_empty(*args, **kwargs)
        torch.empty = mocked_empty

    def tearDown(self):
        torch.empty = self.original_empty

    def test_get_token_bytes_cpu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            token_bytes_list = [0, 1, 2, 3, 4]
            token_bytes_json = os.path.join(temp_dir, "token_bytes.json")
            with open(token_bytes_json, "w") as f:
                json.dump(token_bytes_list, f)

            with patch('prepare.TOKENIZER_DIR', temp_dir):
                result = prepare.get_token_bytes(device="cpu")

                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.dtype, torch.int32)
                self.assertEqual(result.device.type, "cpu")
                self.assertTrue(torch.equal(result, torch.tensor(token_bytes_list, dtype=torch.int32)))

    @patch('torch.cuda.is_available', return_value=False)
    def test_get_token_bytes_mocked_device(self, mock_cuda):
        # We can test with a dummy device like "meta" or just stick to cpu
        with tempfile.TemporaryDirectory() as temp_dir:
            token_bytes_list = [5, 6, 7]
            token_bytes_json = os.path.join(temp_dir, "token_bytes.json")
            with open(token_bytes_json, "w") as f:
                json.dump(token_bytes_list, f)

            with patch('prepare.TOKENIZER_DIR', temp_dir):
                result = prepare.get_token_bytes(device="meta")

                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.dtype, torch.int32)
                self.assertEqual(result.device.type, "meta")
                # meta tensors don't support torch.equal, so we just check shape
                expected = torch.tensor(token_bytes_list, dtype=torch.int32, device="meta")
                self.assertEqual(result.shape, expected.shape)

if __name__ == '__main__':
    unittest.main()
