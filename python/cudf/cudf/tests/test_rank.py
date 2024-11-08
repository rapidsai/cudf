# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from itertools import chain, combinations_with_replacement, product

import numpy as np
import pandas as pd
import pytest

from cudf import DataFrame
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.fixture
def pdf():
    return pd.DataFrame(
        {
            "col1": np.array([5, 4, 3, 5, 8, 5, 2, 1, 6, 6]),
            "col2": np.array(
                [5, 4, np.nan, 5, 8, 5, np.inf, np.nan, 6, -np.inf]
            ),
        },
        index=np.array([5, 4, 3, 2, 1, 6, 7, 8, 9, 10]),
    )


@pytest.mark.parametrize("dtype", ["O", "f8", "i4"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("method", ["average", "min", "max", "first", "dense"])
@pytest.mark.parametrize("na_option", ["keep", "top", "bottom"])
@pytest.mark.parametrize("pct", [True, False])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_rank_all_arguments(
    pdf, dtype, ascending, method, na_option, pct, numeric_only
):
    if method == "first" and dtype == "O":
        # not supported by pandas
        return

    pdf = pdf.copy(deep=True)  # for parallel pytest
    if numeric_only:
        pdf["str"] = np.array(
            ["a", "b", "c", "d", "e", "1", "2", "3", "4", "5"]
        )
    gdf = DataFrame.from_pandas(pdf)

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


def test_rank_error_arguments(pdf):
    gdf = DataFrame.from_pandas(pdf)

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


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
@pytest.mark.parametrize(
    "elem,dtype",
    list(
        product(
            combinations_with_replacement(
                [
                    np.full((3,), np.nan),
                    100 * np.random.default_rng(seed=0).random(10),
                    np.full((3,), np.inf),
                    np.full((3,), -np.inf),
                ],
                4,
            ),
            [np.int32, np.int64, np.float32, np.float64],
        )
    ),
)
def test_series_rank_combinations(elem, dtype):
    aa = np.fromiter(chain.from_iterable(elem), np.float64).astype(dtype)
    gdf = DataFrame({"a": aa})
    df = pd.DataFrame({"a": aa})
    ranked_gs = gdf["a"].rank(method="first")
    ranked_ps = df["a"].rank(method="first")
    # Check
    assert_eq(ranked_ps, ranked_gs)
