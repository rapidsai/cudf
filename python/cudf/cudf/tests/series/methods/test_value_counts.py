# Copyright (c) 2025, NVIDIA CORPORATION.
from string import ascii_letters, digits

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.column.column import as_column
from cudf.testing import assert_eq


@pytest.fixture(params=[True, False])
def normalize(request):
    """Argument for value_counts"""
    return request.param


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.date_range("2010-01-01", "2010-02-01"),
        [None, None],
        [pd.Timestamp(2020, 1, 1), pd.NaT],
    ],
)
def test_series_datetime_value_counts(data, normalize, dropna):
    psr = pd.Series(data, dtype="datetime64[ns]")
    gsr = cudf.from_pandas(psr)
    expected = psr.value_counts(dropna=dropna, normalize=normalize)
    got = gsr.value_counts(dropna=dropna, normalize=normalize)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=False)
    assert_eq(
        expected.reset_index(drop=True),
        got.reset_index(drop=True),
        check_dtype=False,
        check_index_type=True,
    )


def test_categorical_value_counts(dropna, normalize):
    num_elements = 20
    rng = np.random.default_rng(seed=12)
    pd_cat = pd.Categorical(
        pd.Series(
            rng.choice(list(ascii_letters + digits), num_elements),
            dtype="category",
        )
    )

    # gdf
    gdf_value_counts = cudf.Series(pd_cat).value_counts(
        dropna=dropna, normalize=normalize
    )

    # pandas
    pdf_value_counts = pd.Series(pd_cat).value_counts(
        dropna=dropna, normalize=normalize
    )

    # verify
    assert_eq(
        pdf_value_counts.sort_index(),
        gdf_value_counts.sort_index(),
        check_dtype=False,
        check_index_type=True,
    )
    assert_eq(
        pdf_value_counts.reset_index(drop=True),
        gdf_value_counts.reset_index(drop=True),
        check_dtype=False,
        check_index_type=True,
    )


def test_series_value_counts(dropna, normalize):
    rng = np.random.default_rng(seed=0)
    size = 10
    arr = rng.integers(low=-1, high=10, size=size)
    mask = arr != -1
    sr = cudf.Series._from_column(
        as_column(arr).set_mask(cudf.Series(mask)._column.as_mask())
    )
    sr.name = "col"

    expect = (
        sr.to_pandas()
        .value_counts(dropna=dropna, normalize=normalize)
        .sort_index()
    )
    got = sr.value_counts(dropna=dropna, normalize=normalize).sort_index()

    assert_eq(expect, got, check_dtype=True, check_index_type=False)


@pytest.mark.parametrize("bins", [1, 3])
def test_series_value_counts_bins(bins):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(bins=bins)
    got = gsr.value_counts(bins=bins)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=True)


@pytest.mark.parametrize("bins", [1, 3])
def test_series_value_counts_bins_dropna(bins, dropna):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, np.nan])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(bins=bins, dropna=dropna)
    got = gsr.value_counts(bins=bins, dropna=dropna)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=True)


def test_series_value_counts_optional_arguments(ascending, dropna, normalize):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, None])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(
        ascending=ascending, dropna=dropna, normalize=normalize
    )
    got = gsr.value_counts(
        ascending=ascending, dropna=dropna, normalize=normalize
    )

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=True)
    assert_eq(
        expected.reset_index(drop=True),
        got.reset_index(drop=True),
        check_dtype=True,
    )


def test_series_categorical_missing_value_count():
    ps = pd.Series(pd.Categorical(list("abcccb"), categories=list("cabd")))
    gs = cudf.from_pandas(ps)

    expected = ps.value_counts()
    actual = gs.value_counts()

    assert_eq(expected, actual, check_dtype=False)


def test_numeric_alpha_value_counts():
    pdf = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6, 1, 2, 4] * 10,
            "alpha": ["u", "h", "d", "a", "m", "u", "h", "d", "a"] * 10,
        }
    )

    gdf = cudf.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6, 1, 2, 4] * 10,
            "alpha": ["u", "h", "d", "a", "m", "u", "h", "d", "a"] * 10,
        }
    )

    assert_eq(
        pdf.numeric.value_counts().sort_index(),
        gdf.numeric.value_counts().sort_index(),
        check_dtype=False,
    )
    assert_eq(
        pdf.alpha.value_counts().sort_index(),
        gdf.alpha.value_counts().sort_index(),
        check_dtype=False,
    )
