# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("method", ["average", "min", "max", "first", "dense"])
@pytest.mark.parametrize("na_option", ["keep", "top", "bottom"])
@pytest.mark.parametrize("pct", [True, False])
def test_rank_all_arguments(ascending, method, na_option, pct, numeric_only):
    pdf = pd.DataFrame(
        {
            "col1": np.array([5, 4, 3, 5, 8, 5, 2, 1, 6, 6]),
            "col2": np.array(
                [5, 4, np.nan, 5, 8, 5, np.inf, np.nan, 6, -np.inf]
            ),
        },
        index=np.array([5, 4, 3, 2, 1, 6, 7, 8, 9, 10]),
    )

    if numeric_only:
        pdf["str"] = np.array(
            ["a", "b", "c", "d", "e", "1", "2", "3", "4", "5"]
        )
    gdf = cudf.DataFrame(pdf)

    kwargs = {
        "method": method,
        "na_option": na_option,
        "ascending": ascending,
        "pct": pct,
        "numeric_only": numeric_only,
    }

    # Series
    assert_eq(gdf["col1"].rank(**kwargs), pdf["col1"].rank(**kwargs))
    assert_eq(gdf["col2"].rank(**kwargs), pdf["col2"].rank(**kwargs))
    if numeric_only:
        assert_exceptions_equal(
            lfunc=pdf["str"].rank,
            rfunc=gdf["str"].rank,
            lfunc_args_and_kwargs=(
                [],
                kwargs,
            ),
            rfunc_args_and_kwargs=(
                [],
                kwargs,
            ),
        )

    actual = gdf.rank(**kwargs)
    expected = pdf.rank(**kwargs)

    assert_eq(expected, actual)


def test_rank_error_arguments():
    pdf = pd.DataFrame(
        {
            "col1": np.array([5, 4, 3, 5, 8, 5, 2, 1, 6, 6]),
            "col2": np.array(
                [5, 4, np.nan, 5, 8, 5, np.inf, np.nan, 6, -np.inf]
            ),
        },
        index=np.array([5, 4, 3, 2, 1, 6, 7, 8, 9, 10]),
    )
    gdf = cudf.DataFrame(pdf)

    assert_exceptions_equal(
        lfunc=pdf["col1"].rank,
        rfunc=gdf["col1"].rank,
        lfunc_args_and_kwargs=(
            [],
            {
                "method": "randomname",
                "na_option": "keep",
                "ascending": True,
                "pct": True,
            },
        ),
        rfunc_args_and_kwargs=(
            [],
            {
                "method": "randomname",
                "na_option": "keep",
                "ascending": True,
                "pct": True,
            },
        ),
    )

    assert_exceptions_equal(
        lfunc=pdf["col1"].rank,
        rfunc=gdf["col1"].rank,
        lfunc_args_and_kwargs=(
            [],
            {
                "method": "first",
                "na_option": "randomname",
                "ascending": True,
                "pct": True,
            },
        ),
        rfunc_args_and_kwargs=(
            [],
            {
                "method": "first",
                "na_option": "randomname",
                "ascending": True,
                "pct": True,
            },
        ),
    )
