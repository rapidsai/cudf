# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.io import make_partitioned_source


@pytest.mark.parametrize("fmt", ["csv", "ndjson", "parquet"])
@pytest.mark.parametrize("use_str_path", [True, False])
def test_make_source_single_file(tmp_path, fmt, use_str_path):
    df = pl.DataFrame({"a": [1, 2], "b": ["foo", "bar"]})
    path = tmp_path / f"df.{fmt}"
    make_partitioned_source(df, str(path) if use_str_path else path, fmt)

    assert path.exists()
    assert path.is_file()


@pytest.mark.parametrize("fmt", ["csv", "ndjson", "parquet"])
@pytest.mark.parametrize("n_files", [2, 5])
def test_make_source_multiple_files(tmp_path, fmt, n_files):
    df = pl.DataFrame({"a": list(range(100)), "b": ["x"] * 100})
    make_partitioned_source(df, tmp_path, fmt, n_files=n_files)

    for i, file in enumerate(sorted(tmp_path.iterdir())):
        assert file.exists()
        assert file.is_file()
        assert file.name == f"part.{i}.{fmt}"


def test_make_source_invalid_format(tmp_path):
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Unsupported format: foo"):
        make_partitioned_source(df, tmp_path / "bad.foo", "foo")
