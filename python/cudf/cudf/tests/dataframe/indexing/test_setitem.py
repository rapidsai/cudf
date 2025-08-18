# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_setitem_datetime():
    df = cudf.DataFrame()
    df["date"] = pd.date_range("20010101", "20010105").values
    assert df.date.dtype.kind == "M"


@pytest.mark.parametrize("scalar", ["a", None])
def test_string_set_scalar(scalar):
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    pdf["b"] = "a"
    gdf["b"] = "a"

    assert_eq(pdf["b"], gdf["b"])
    assert_eq(pdf, gdf)
