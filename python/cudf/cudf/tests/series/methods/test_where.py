# Copyright (c) 2025, NVIDIA CORPORATION.
import re

import pytest

import cudf


def test_series_where_mixed_dtypes_error():
    s = cudf.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "cudf does not support mixed types, please type-cast "
            "the column of dataframe/series and other "
            "to same dtypes."
        ),
    ):
        s.where([True, False, True], [1, 2, 3])


def test_series_where_mixed_bool_dtype():
    s = cudf.Series([True, False, True])
    with pytest.raises(TypeError):
        s.where(~s, 10)
