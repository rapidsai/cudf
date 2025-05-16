# Copyright (c) 2023-2025, NVIDIA CORPORATION.
from io import BytesIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "index",
    [range(1, 11), list(range(1, 11)), range(1, 11)[::2]],
    ids=["RangeIndex", "IntIndex", "StridedRange"],
)
@pytest.mark.parametrize("write_index", [False, True, None])
@pytest.mark.parametrize("empty", [False, True], ids=["nonempty", "empty"])
def test_dataframe_parquet_roundtrip(index, write_index, empty):
    if empty:
        data = {}
    else:
        data = {"a": [i * 2 for i in index]}
    df = cudf.DataFrame(data=data, index=index)
    pf = pd.DataFrame(data=data, index=index)
    gpu_buf = BytesIO()
    cpu_buf = BytesIO()

    df.to_parquet(gpu_buf, index=write_index)
    pf.to_parquet(cpu_buf, index=write_index)
    gpu_table = pq.read_table(gpu_buf)
    cpu_table = pq.read_table(cpu_buf)
    metadata_equal = (
        gpu_table.schema.pandas_metadata == cpu_table.schema.pandas_metadata
    )
    assert metadata_equal

    gpu_read = cudf.read_parquet(gpu_buf)
    cpu_read = cudf.read_parquet(cpu_buf)
    assert_eq(gpu_read, cpu_read)


@pytest.mark.parametrize("preserve_index", [False, True, None])
def test_dataframe_to_arrow_preserve_index(preserve_index):
    df = cudf.DataFrame({"x": ["cat", "dog"] * 5})
    pf = df.to_pandas()
    expect = pa.Table.from_pandas(pf, preserve_index=preserve_index).schema
    got = df.to_arrow(preserve_index=preserve_index).schema
    assert expect == got
