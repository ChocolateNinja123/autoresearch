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
