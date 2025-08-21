# Copyright (c) 2025, NVIDIA CORPORATION.
import datetime
import decimal

import pandas as pd
import pyarrow as pa
import pytest

import cudf


def test_list_to_pandas_nullable_true():
    df = cudf.DataFrame({"a": cudf.Series([[1, 2, 3]])})
    with pytest.raises(NotImplementedError):
        df.to_pandas(nullable=True)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
        [1],
        decimal.Decimal("1.0"),
    ],
)
def test_dataframe_to_pandas_arrow_type(scalar):
    pa_array = pa.array([scalar, None])
    df = cudf.DataFrame({"a": pa_array})
    result = df.to_pandas(arrow_type=True)
    expected = pd.DataFrame({"a": pd.arrays.ArrowExtensionArray(pa_array)})
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
        [1],
        decimal.Decimal("1.0"),
    ],
)
def test_dataframe_to_pandas_arrow_type_nullable_raises(scalar):
    pa_array = pa.array([scalar, None])
    df = cudf.DataFrame({"a": pa_array})
    with pytest.raises(ValueError):
        df.to_pandas(nullable=True, arrow_type=True)
