# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_groupby_results_equal
from cudf.testing.dataset_generator import rand_dataframe


def test_groupby_select_then_ffill():
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [1, None, None, 2, None],
            "c": [3, None, None, 4, None],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("a")["c"].ffill()
    actual = gdf.groupby("a")["c"].ffill()

    assert_groupby_results_equal(expected, actual)


@pytest.mark.xfail(
    reason="decimal64 .to_pandas() fillna null with None instead of NaN"
)
def test_groupby_ffill_multi_value():
    nelem = 20
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ms]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {"dtype": "decimal64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "str", "null_frequency": 0.4, "cardinality": 10},
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5", "6"]
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(key_col).ffill()
    got = gdf.groupby(key_col).ffill()

    assert_groupby_results_equal(expect[value_cols], got[value_cols])
