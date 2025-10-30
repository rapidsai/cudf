# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "decimals",
    [
        -3,
        0,
        5,
        pd.Series([1, -2, 3], index=["ints", "ints_with_na", "ints_same"]),
        cudf.Series([-1, 2, 12], index=["ints", "ints_with_na", "ints_same"]),
        {"ints": -1, "ints_with_na": 2, "ints_same": 3},
    ],
)
def test_dataframe_round(decimals):
    rng = np.random.default_rng(seed=0)
    gdf = cudf.DataFrame(
        {
            "ints": rng.integers(-1000, 1000, 10),
            "ints_with_na": [
                14123,
                2343,
                None,
                0,
                -8302,
                None,
                94313,
                None,
                -8029,
                None,
            ],
            "ints_same": np.repeat([123456], 10),
            "bools": rng.choice([True, None, False], 10),
            "strings": rng.choice(["abc", "xyz", None], 10),
            "struct": rng.choice([{"abc": 1}, {"xyz": 2}, None], 10),
            "list": [[1], [2], None, [4], [3]] * 2,
        }
    )
    pdf = gdf.to_pandas()

    if isinstance(decimals, cudf.Series):
        pdecimals = decimals.to_pandas()
    else:
        pdecimals = decimals

    result = gdf.round(decimals)
    expected = pdf.round(pdecimals)

    assert_eq(result, expected)


def test_dataframe_round_decimal():
    """Test rounding with decimal columns - separate test since pandas handles differently"""
    from decimal import Decimal

    gdf = cudf.DataFrame(
        {
            "dec1": cudf.Series(
                [Decimal("123.456"), Decimal("-987.654"), Decimal("0.123")],
                dtype=cudf.Decimal64Dtype(precision=6, scale=3),
            ),
            "dec2": cudf.Series(
                [Decimal("1234.5678"), Decimal("-9876.5432"), None],
                dtype=cudf.Decimal64Dtype(precision=8, scale=4),
            ),
        }
    )

    # Round to 1 decimal place for dec1, 2 decimal places for dec2
    result = gdf.round({"dec1": 1, "dec2": 2})

    # Verify the rounding worked
    assert isinstance(result["dec1"].dtype, cudf.Decimal64Dtype)
    assert isinstance(result["dec2"].dtype, cudf.Decimal64Dtype)

    # Check the actual rounded values
    result_dec1 = result["dec1"].to_pandas().tolist()
    result_dec2 = result["dec2"].to_pandas().tolist()

    # dec1 rounded to 1 decimal place: 123.456 -> 123.5, -987.654 -> -987.7, 0.123 -> 0.1
    assert result_dec1[0] == Decimal("123.5")
    assert result_dec1[1] == Decimal("-987.7")
    assert result_dec1[2] == Decimal("0.1")

    # dec2 rounded to 2 decimal places: 1234.5678 -> 1234.57, -9876.5432 -> -9876.54, None -> None
    assert result_dec2[0] == Decimal("1234.57")
    assert result_dec2[1] == Decimal("-9876.54")
    assert result_dec2[2] is None


def test_dataframe_round_dict_decimal_validation():
    df = cudf.DataFrame({"A": [123, 456], "B": [789, 12]})
    with pytest.raises(TypeError):
        df.round({"A": 1, "B": 0.5})
