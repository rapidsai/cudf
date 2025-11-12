# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd

import cudf
from cudf.testing import assert_groupby_results_equal


def test_size_as_index_false():
    df = pd.DataFrame({"a": [1, 2, 1], "b": [1, 2, 3]}, columns=["a", "b"])
    expected = df.groupby("a", as_index=False).size()
    result = cudf.from_pandas(df).groupby("a", as_index=False).size()
    assert_groupby_results_equal(result, expected, as_index=False, by="a")


def test_size_series_with_name():
    ser = pd.Series(range(3), name="foo")
    expected = ser.groupby(ser).size()
    result = cudf.from_pandas(ser).groupby(ser).size()
    assert_groupby_results_equal(result, expected)
