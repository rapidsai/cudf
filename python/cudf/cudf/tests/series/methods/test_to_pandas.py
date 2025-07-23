# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "sr_data,expected_psr",
    [
        (
            pa.array([1, 2, None, 3], type=pa.uint8()),
            pd.Series([1, 2, None, 3], dtype=pd.UInt8Dtype()),
        ),
        (
            pa.array([23, None, None, 32], type=pa.uint16()),
            pd.Series([23, None, None, 32], dtype=pd.UInt16Dtype()),
        ),
        (
            pa.array([None, 123, None, 1], type=pa.uint32()),
            pd.Series([None, 123, None, 1], dtype=pd.UInt32Dtype()),
        ),
        (
            pa.array([234, 2323, 23432, None, None, 224], type=pa.uint64()),
            pd.Series(
                [234, 2323, 23432, None, None, 224], dtype=pd.UInt64Dtype()
            ),
        ),
        (
            pa.array([-10, 1, None, -1, None, 3], type=pa.int8()),
            pd.Series([-10, 1, None, -1, None, 3], dtype=pd.Int8Dtype()),
        ),
        (
            pa.array([111, None, 222, None, 13], type=pa.int16()),
            pd.Series([111, None, 222, None, 13], dtype=pd.Int16Dtype()),
        ),
        (
            pa.array([11, None, 22, 33, None, 2, None, 3], type=pa.int32()),
            pd.Series(
                [11, None, 22, 33, None, 2, None, 3], dtype=pd.Int32Dtype()
            ),
        ),
        (
            pa.array(
                [32431, None, None, 32322, 0, 10, -32324, None],
                type=pa.int64(),
            ),
            pd.Series(
                [32431, None, None, 32322, 0, 10, -32324, None],
                dtype=pd.Int64Dtype(),
            ),
        ),
        (
            pa.array(
                [True, None, False, None, False, True, True, False],
                type=pa.bool_(),
            ),
            pd.Series(
                [True, None, False, None, False, True, True, False],
                dtype=pd.BooleanDtype(),
            ),
        ),
        (
            pa.array(
                [
                    "abc",
                    "a",
                    None,
                    "hello world",
                    "foo buzz",
                    "",
                    None,
                    "rapids ai",
                ],
                type=pa.string(),
            ),
            pd.Series(
                [
                    "abc",
                    "a",
                    None,
                    "hello world",
                    "foo buzz",
                    "",
                    None,
                    "rapids ai",
                ],
                dtype=pd.StringDtype(),
            ),
        ),
        (
            pa.array(
                [1, 2, None, 10.2, None],
                type=pa.float32(),
            ),
            pd.Series(
                [1, 2, None, 10.2, None],
                dtype=pd.Float32Dtype(),
            ),
        ),
    ],
)
def test_series_to_pandas_nullable_dtypes(sr_data, expected_psr):
    sr = cudf.Series(sr_data)
    actual_psr = sr.to_pandas(nullable=True)

    assert_eq(actual_psr, expected_psr)
