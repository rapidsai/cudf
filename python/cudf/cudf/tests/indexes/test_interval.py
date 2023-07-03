# Copyright (c) 2023, NVIDIA CORPORATION.
import pandas as pd
import pyarrow as pa

import cudf
from cudf.testing._utils import assert_eq


def test_interval_constructor_default_closed():
    idx = cudf.IntervalIndex([pd.Interval(0, 1)])
    assert idx.closed == "right"
    assert idx.dtype.closed == "right"


def test_interval_to_arrow():
    expect = pa.Array.from_pandas(pd.IntervalIndex([pd.Interval(0, 1)]))
    got = cudf.IntervalIndex([pd.Interval(0, 1)]).to_arrow()
    assert_eq(expect, got)
