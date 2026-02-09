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
        pd.Series(
            [1, 4, 3, -6],
            index=["floats", "ints", "floats_with_nan", "floats_same"],
        ),
        cudf.Series(
            [-4, -2, 12], index=["ints", "floats_with_nan", "floats_same"]
        ),
        {"floats": -1, "ints": 15, "floats_will_nan": 2},
    ],
)
def test_dataframe_round(decimals):
    rng = np.random.default_rng(seed=0)
    gdf = cudf.DataFrame(
        {
            "floats": np.arange(0.5, 10.5, 1),
            "ints": rng.normal(-100, 100, 10),
            "floats_with_na": np.array(
                [
                    14.123,
                    2.343,
                    np.nan,
                    0.0,
                    -8.302,
                    np.nan,
                    94.313,
                    None,
                    -8.029,
                    np.nan,
                ]
            ),
            "floats_same": np.repeat([-0.6459412758761901], 10),
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


def test_dataframe_round_dict_decimal_validation():
    df = cudf.DataFrame({"A": [0.12], "B": [0.13]})
    with pytest.raises(TypeError):
        df.round({"A": 1, "B": 0.5})
