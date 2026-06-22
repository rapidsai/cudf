# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "data",
    [
        pd.MultiIndex.from_arrays(
            [[1, 2, 3], ["red", "blue", "green"]], names=("number", "color")
        ),
        pd.MultiIndex.from_arrays([[], []], names=("number", "color")),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 10, 100], ["red", "blue", "green", "pink", "white"]],
            names=("number", "color"),
        ),
        pd.MultiIndex.from_product(
            [[0, 1], ["red", "blue", "green"]], names=("number", "color")
        ),
    ],
)
@pytest.mark.parametrize(
    "values,level,err",
    [
        ([(1, "red"), (2, "blue"), (0, "green")], None, None),
        (["red", "orange", "yellow"], "color", None),
        (["red", "white", "yellow"], "color", None),
        ([0, 1, 2, 10, 11, 15], "number", None),
        ([0, 1, 2, 10, 11, 15], None, TypeError),
        (pd.Series([0, 1, 2, 10, 11, 15]), None, TypeError),
        (pd.Index([0, 1, 2, 10, 11, 15]), None, TypeError),
        (pd.Index([0, 1, 2, 8, 11, 15]), "number", None),
        (pd.Index(["red", "white", "yellow"]), "color", None),
        ([(1, "red"), (3, "red")], None, None),
        (((1, "red"), (3, "red")), None, None),
        (
            pd.MultiIndex.from_arrays(
                [[1, 2, 3], ["red", "blue", "green"]],
                names=("number", "color"),
            ),
            None,
            None,
        ),
        (
            pd.MultiIndex.from_arrays([[], []], names=("number", "color")),
            None,
            None,
        ),
        (
            pd.MultiIndex.from_arrays(
                [
                    [1, 2, 3, 10, 100],
                    ["red", "blue", "green", "pink", "white"],
                ],
                names=("number", "color"),
            ),
            None,
            None,
        ),
    ],
)
def test_isin_multiindex(data, values, level, err):
    pmdx = data
    gmdx = cudf.from_pandas(data)

    if err is None:
        expected = pmdx.isin(values, level=level)
        if isinstance(values, pd.MultiIndex):
            values = cudf.from_pandas(values)
        got = gmdx.isin(values, level=level)

        assert_eq(got, expected)
    else:
        assert_exceptions_equal(
            lfunc=pmdx.isin,
            rfunc=gmdx.isin,
            lfunc_args_and_kwargs=([values], {"level": level}),
            rfunc_args_and_kwargs=([values], {"level": level}),
            check_exception_type=False,
        )
