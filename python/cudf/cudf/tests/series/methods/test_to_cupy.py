# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("has_nulls", [False, True])
@pytest.mark.parametrize("use_na_value", [False, True])
def test_series_to_cupy(
    numeric_and_bool_types_as_str, has_nulls, use_na_value
):
    size = 10
    if numeric_and_bool_types_as_str == "bool":
        np_data = np.array([True, False] * (size // 2), dtype=bool)
    else:
        np_data = np.arange(size, dtype=numeric_and_bool_types_as_str)

    if has_nulls:
        np_data = np_data.astype("object")
        np_data[::2] = None

    sr = cudf.Series(np_data, dtype=numeric_and_bool_types_as_str)

    if not has_nulls:
        assert_eq(sr.values, cp.asarray(sr))
        return

    if has_nulls and not use_na_value:
        if numeric_and_bool_types_as_str == "bool":
            with pytest.raises(ValueError, match="Column must have no nulls"):
                sr.to_cupy()
        else:
            result = sr.to_cupy()
            expected = (
                sr.astype(
                    "float32"
                    if numeric_and_bool_types_as_str == "float32"
                    else "float64"
                )
                .fillna(np.nan)
                .to_cupy()
            )
            assert_eq(result, expected)
        return

    na_value = {
        "bool": False,
        "float32": 0.0,
        "float64": 0.0,
    }.get(numeric_and_bool_types_as_str, 0)
    expected = cp.asarray(sr.fillna(na_value)) if has_nulls else cp.asarray(sr)
    assert_eq(sr.to_cupy(na_value=na_value), expected)
