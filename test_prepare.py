import os
import pytest
from unittest.mock import patch

import prepare


def test_list_parquet_files_normal(tmp_path):
    """Test that list_parquet_files returns sorted .parquet files and ignores others."""
    # Create dummy files
    (tmp_path / "shard_00002.parquet").touch()
    (tmp_path / "shard_00000.parquet").touch()
    (tmp_path / "shard_00001.parquet").touch()
    (tmp_path / "shard_00003.parquet.tmp").touch()  # Should be ignored
    (tmp_path / "other_file.txt").touch()           # Should be ignored
    (tmp_path / "not_a_parquet").touch()            # Should be ignored

    expected = [
        os.path.join(str(tmp_path), "shard_00000.parquet"),
        os.path.join(str(tmp_path), "shard_00001.parquet"),
        os.path.join(str(tmp_path), "shard_00002.parquet"),
    ]

    with patch("prepare.DATA_DIR", str(tmp_path)):
        result = prepare.list_parquet_files()
        assert result == expected


def test_list_parquet_files_empty(tmp_path):
    """Test that list_parquet_files returns an empty list when no .parquet files exist."""
    with patch("prepare.DATA_DIR", str(tmp_path)):
        result = prepare.list_parquet_files()
        assert result == []


def test_list_parquet_files_only_tmp_and_others(tmp_path):
    """Test that list_parquet_files correctly filters out .tmp files even if they have .parquet in name."""
    (tmp_path / "shard_00001.parquet.tmp").touch()
    (tmp_path / "random.txt").touch()

    with patch("prepare.DATA_DIR", str(tmp_path)):
        result = prepare.list_parquet_files()
        assert result == []

@patch('os.listdir')
@patch('os.path.join')
def test_list_parquet_files_with_mocks(mock_join, mock_listdir):
    """Test list_parquet_files using mocks for os.listdir and os.path.join."""
    mock_listdir.return_value = [
        "shard_00002.parquet",
        "random.txt",
        "shard_00000.parquet",
        "shard_00001.parquet",
        "ignored.parquet.tmp",
        "not_a_parquet.parquet_but_wrong",
    ]

    mock_join.side_effect = lambda *args: "/".join(args)

    with patch("prepare.DATA_DIR", "mock_dir"):
        result = prepare.list_parquet_files()

    expected = [
        "mock_dir/shard_00000.parquet",
        "mock_dir/shard_00001.parquet",
        "mock_dir/shard_00002.parquet",
    ]
    assert result == expected

    mock_listdir.assert_called_once_with("mock_dir")

    assert mock_join.call_count == 3
    # Check that they were called in the expected sorted order
    mock_join.assert_any_call("mock_dir", "shard_00000.parquet")
    mock_join.assert_any_call("mock_dir", "shard_00001.parquet")
    mock_join.assert_any_call("mock_dir", "shard_00002.parquet")

@patch('os.listdir')
def test_list_parquet_files_dir_not_found(mock_listdir):
    """Test list_parquet_files handles FileNotFoundError gracefully (if it bubbles up or is tested to raise it)."""
    # The function prepare.list_parquet_files() doesn't currently catch FileNotFoundError,
    # so we expect it to raise it.
    mock_listdir.side_effect = FileNotFoundError("No such file or directory: 'mock_dir'")

    with patch("prepare.DATA_DIR", "mock_dir"):
        with pytest.raises(FileNotFoundError):
            prepare.list_parquet_files()

import torch

def test_get_token_bytes_normal(tmp_path):
    """Test get_token_bytes loads from the correct path and returns a tensor on the correct device."""
    import torch

    expected_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cpu")

    # Create the token_bytes.pt file in the mocked TOKENIZER_DIR
    mock_tokenizer_dir = tmp_path / "tokenizer"
    mock_tokenizer_dir.mkdir()
    token_bytes_path = mock_tokenizer_dir / "token_bytes.pt"

    torch.save(expected_tensor, token_bytes_path)

    with patch("prepare.TOKENIZER_DIR", str(mock_tokenizer_dir)):
        result = prepare.get_token_bytes(device="cpu")

        # Verify it returns a torch.Tensor
        assert isinstance(result, torch.Tensor)
        # Verify correct dtype
        assert result.dtype == torch.int32
        # Verify correct device
        assert result.device.type == "cpu"
        # Verify the contents match
        assert result.tolist() == expected_tensor.tolist()

def test_get_token_bytes_file_not_found(tmp_path):
    """Test get_token_bytes raises FileNotFoundError if token_bytes.pt is missing."""
    mock_tokenizer_dir = tmp_path / "tokenizer"
    mock_tokenizer_dir.mkdir()

    with patch("prepare.TOKENIZER_DIR", str(mock_tokenizer_dir)):
        with pytest.raises(FileNotFoundError):
            prepare.get_token_bytes(device="cpu")

@patch("prepare.torch.load")
@patch("builtins.open")
def test_get_token_bytes_mocked(mock_open, mock_torch_load):
    """Test get_token_bytes using mocks to verify internal calls."""
    mock_torch_load.return_value = "mock_tensor"

    with patch("prepare.TOKENIZER_DIR", "/mock/dir"):
        result = prepare.get_token_bytes(device="cuda")

    expected_path = os.path.join("/mock/dir", "token_bytes.pt")
    mock_open.assert_called_once_with(expected_path, "rb")
    mock_torch_load.assert_called_once_with(mock_open.return_value.__enter__(), map_location="cuda")
    assert result == "mock_tensor"

import requests
from unittest.mock import MagicMock

def test_download_single_shard_exists(tmp_path):
    """Test when the file already exists, it returns True immediately."""
    index = 42
    filename = f"shard_{index:05d}.parquet"
    filepath = tmp_path / filename
    filepath.touch()

    with patch("prepare.DATA_DIR", str(tmp_path)):
        with patch("prepare.requests.get") as mock_get:
            result = prepare.download_single_shard(index)
            assert result is True
            mock_get.assert_not_called()

@patch("prepare.time.sleep")
def test_download_single_shard_success_first_try(mock_sleep, tmp_path):
    """Test successful download on the first attempt."""
    index = 1
    filename = f"shard_{index:05d}.parquet"
    filepath = tmp_path / filename

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]

    with patch("prepare.DATA_DIR", str(tmp_path)):
        with patch("prepare.requests.get") as mock_get:
            mock_get.return_value = mock_response

            result = prepare.download_single_shard(index)

            assert result is True
            assert mock_get.call_count == 1
            mock_sleep.assert_not_called()

            # Check file contents
            assert filepath.exists()
            assert filepath.read_bytes() == b"chunk1chunk2"

            # Ensure tmp file is gone
            assert not (tmp_path / (filename + ".tmp")).exists()

@patch("prepare.time.sleep")
def test_download_single_shard_success_after_retries(mock_sleep, tmp_path):
    """Test successful download after initial failures."""
    index = 2
    filename = f"shard_{index:05d}.parquet"
    filepath = tmp_path / filename

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [b"success_data"]

    with patch("prepare.DATA_DIR", str(tmp_path)):
        with patch("prepare.requests.get") as mock_get:
            # First two raise, third succeeds
            mock_get.side_effect = [
                requests.RequestException("Network Error"),
                IOError("File Error"),
                mock_response
            ]

            result = prepare.download_single_shard(index)

            assert result is True
            assert mock_get.call_count == 3

            # Check sleep was called correctly for attempts 1 and 2
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(2)  # attempt 1: 2^1 = 2
            mock_sleep.assert_any_call(4)  # attempt 2: 2^2 = 4

            # Check file contents
            assert filepath.exists()
            assert filepath.read_bytes() == b"success_data"

@patch("prepare.time.sleep")
def test_download_single_shard_failure(mock_sleep, tmp_path):
    """Test that it returns False after max_attempts."""
    index = 3
    filename = f"shard_{index:05d}.parquet"
    filepath = tmp_path / filename

    with patch("prepare.DATA_DIR", str(tmp_path)):
        with patch("prepare.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Persistent Error")

            result = prepare.download_single_shard(index)

            assert result is False
            assert mock_get.call_count == 5

            # Check sleep was called correctly
            assert mock_sleep.call_count == 4
            mock_sleep.assert_any_call(2)
            mock_sleep.assert_any_call(4)
            mock_sleep.assert_any_call(8)
            mock_sleep.assert_any_call(16)

            # Ensure files are cleaned up
            assert not filepath.exists()
            assert not (tmp_path / (filename + ".tmp")).exists()
