# Copyright (c) 2023-2025, NVIDIA CORPORATION.

import pyarrow as pa
import pytest

import cudf


@pytest.mark.parametrize("preserve_index", [False, True, None])
def test_dataframe_to_arrow_preserve_index(preserve_index):
    df = cudf.DataFrame({"x": ["cat", "dog"] * 5})
    pf = df.to_pandas()
    expect = pa.Table.from_pandas(pf, preserve_index=preserve_index).schema
    got = df.to_arrow(preserve_index=preserve_index).schema
    assert expect == got
