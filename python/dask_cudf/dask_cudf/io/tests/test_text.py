# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import dask.dataframe as dd

import cudf

import dask_cudf

cur_dir = os.path.dirname(__file__)
text_file = os.path.join(cur_dir, "data/text/sample.pgn")


@pytest.mark.parametrize("file", [text_file, [text_file]])
@pytest.mark.parametrize("chunksize", [12, "50 B", None])
def test_read_text(file, chunksize):
    df1 = cudf.read_text(text_file, delimiter='"]')
    df2 = dask_cudf.read_text(file, chunksize=chunksize, delimiter='"]')
    dd.assert_eq(df1, df2, check_index=False)


@pytest.mark.parametrize("offset", [0, 100, 250, 500, 1000])
@pytest.mark.parametrize("size", [64, 128, 256])
def test_read_text_byte_range(offset, size):
    df1 = cudf.read_text(text_file, delimiter=".", byte_range=(offset, size))
    df2 = dask_cudf.read_text(
        text_file, chunksize=None, delimiter=".", byte_range=(offset, size)
    )
    dd.assert_eq(df1, df2, check_index=False)


def test_deprecated_api_paths():
    # Encourage top-level read_text import only
    df = cudf.read_text(text_file, delimiter=".")
    with pytest.warns(match="dask_cudf.io.read_text is now deprecated"):
        df2 = dask_cudf.io.read_text(text_file, delimiter=".")
    dd.assert_eq(df, df2, check_divisions=False)
