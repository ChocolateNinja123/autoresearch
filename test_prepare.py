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
