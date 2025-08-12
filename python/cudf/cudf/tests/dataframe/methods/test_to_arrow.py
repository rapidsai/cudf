# Copyright (c) 2023-2025, NVIDIA CORPORATION.

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("preserve_index", [False, True, None])
def test_dataframe_to_arrow_preserve_index(preserve_index):
    df = cudf.DataFrame({"x": ["cat", "dog"] * 5})
    pf = df.to_pandas()
    expect = pa.Table.from_pandas(pf, preserve_index=preserve_index).schema
    got = df.to_arrow(preserve_index=preserve_index).schema
    assert expect == got


def test_datetime_to_arrow(datetime_types_as_str):
    data = pd.date_range("2000-01-01", "2000-01-02", freq="3600s")
    gdf = cudf.DataFrame({"timestamp": data.astype(datetime_types_as_str)})
    assert_eq(
        gdf, cudf.DataFrame.from_arrow(gdf.to_arrow(preserve_index=False))
    )
