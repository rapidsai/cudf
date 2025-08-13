# Copyright (c) 2023-2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq, assert_groupby_results_equal
from cudf.testing._utils import expect_warning_if


def test_rank_return_type_compatible_mode():
    # in compatible mode, rank() always returns floats
    df = cudf.DataFrame({"a": range(10), "b": [0] * 10}, index=[0] * 10)
    pdf = df.to_pandas()
    expect = pdf.groupby("b").get_group(0)
    result = df.groupby("b").get_group(0)
    assert_eq(expect, result)


@pytest.mark.parametrize(
    "pdf, group, name, obj",
    [
        (
            pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]}),
            "X",
            "A",
            None,
        ),
        (
            pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]}),
            "X",
            "B",
            None,
        ),
        (
            pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]}),
            "X",
            "A",
            pd.DataFrame({"a": [1, 2, 4, 5, 10, 11]}),
        ),
        (
            pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]}),
            "Y",
            1,
            pd.DataFrame({"a": [1, 2, 4, 5, 10, 11]}),
        ),
        (
            pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]}),
            "Y",
            3,
            pd.DataFrame({"a": [1, 2, 0, 11]}),
        ),
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warnings only given on newer versions.",
)
def test_groupby_get_group(pdf, group, name, obj):
    gdf = cudf.from_pandas(pdf)

    if isinstance(obj, pd.DataFrame):
        gobj = cudf.from_pandas(obj)
    else:
        gobj = obj

    pgb = pdf.groupby(group)
    ggb = gdf.groupby(group)
    with expect_warning_if(obj is not None):
        expected = pgb.get_group(name=name, obj=obj)
    with expect_warning_if(obj is not None):
        actual = ggb.get_group(name=name, obj=gobj)

    assert_groupby_results_equal(expected, actual)

    expected = pdf.iloc[pgb.indices.get(name)]
    actual = gdf.iloc[ggb.indices.get(name)]

    assert_eq(expected, actual)
