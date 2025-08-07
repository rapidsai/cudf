# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_GE_220,
    PANDAS_GT_214,
)
from cudf.testing import assert_eq
from cudf.testing._utils import (
    expect_warning_if,
)


@pytest.mark.parametrize(
    "gsr_data, dtype",
    [
        [[5, 1, 2, 3, None, 243, None, 4], None],
        [["one", "two", "three", None, "one"], "category"],
        [[*list(range(400)), None], None],
    ],
)
@pytest.mark.parametrize(
    "to_replace,value",
    [
        (0, 5),
        ("one", "two"),
        ("one", "five"),
        ("abc", "hello"),
        ([0, 1], [5, 6]),
        ([22, 323, 27, 0], -1),
        ([1, 2, 3], cudf.Series([10, 11, 12])),
        (cudf.Series([1, 2, 3]), None),
        ({1: 10, 2: 22}, None),
        (np.inf, 4),
    ],
)
def test_series_replace_all(gsr_data, dtype, to_replace, value):
    gsr = cudf.Series(gsr_data, dtype=dtype)
    psr = gsr.to_pandas()

    gd_to_replace = to_replace
    if isinstance(to_replace, cudf.Series):
        pd_to_replace = to_replace.to_pandas()
    else:
        pd_to_replace = to_replace

    gd_value = value
    if isinstance(value, cudf.Series):
        pd_value = value.to_pandas()
    else:
        pd_value = value

    expect_warn = (
        isinstance(gsr.dtype, cudf.CategoricalDtype)
        and isinstance(gd_to_replace, str)
        and gd_to_replace == "one"
    )
    with expect_warning_if(expect_warn):
        actual = gsr.replace(to_replace=gd_to_replace, value=gd_value)
    with expect_warning_if(expect_warn and PANDAS_GE_220):
        if pd_value is None:
            # TODO: Remove this workaround once cudf
            # introduces `no_default` values
            expected = psr.replace(to_replace=pd_to_replace)
        else:
            expected = psr.replace(to_replace=pd_to_replace, value=pd_value)

    assert_eq(
        expected.sort_values().reset_index(drop=True),
        actual.sort_values().reset_index(drop=True),
    )


def test_series_replace():
    a1 = np.array([0, 1, 2, 3, 4])

    # Numerical
    a2 = np.array([5, 1, 2, 3, 4])
    sr1 = cudf.Series(a1)
    sr2 = sr1.replace(0, 5)
    assert_eq(a2, sr2.to_numpy())

    # Categorical
    psr3 = pd.Series(["one", "two", "three"], dtype="category")
    with expect_warning_if(PANDAS_GE_220, FutureWarning):
        psr4 = psr3.replace("one", "two")
    sr3 = cudf.from_pandas(psr3)
    with pytest.warns(FutureWarning):
        sr4 = sr3.replace("one", "two")
    assert_eq(
        psr4.sort_values().reset_index(drop=True),
        sr4.sort_values().reset_index(drop=True),
    )
    with expect_warning_if(PANDAS_GE_220, FutureWarning):
        psr5 = psr3.replace("one", "five")
    with pytest.warns(FutureWarning):
        sr5 = sr3.replace("one", "five")

    assert_eq(psr5, sr5)

    # List input
    a6 = np.array([5, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [5, 6])
    assert_eq(a6, sr6.to_numpy())

    assert_eq(
        sr1.replace([0, 1], [5.5, 6.5]),
        sr1.to_pandas().replace([0, 1], [5.5, 6.5]),
    )

    # Series input
    a8 = np.array([5, 5, 5, 3, 4])
    sr8 = sr1.replace(sr1[:3].to_numpy(), 5)
    assert_eq(a8, sr8.to_numpy())

    # large input containing null
    sr9 = cudf.Series([*list(range(400)), None])
    sr10 = sr9.replace([22, 323, 27, 0], None)
    assert sr10.null_count == 5
    assert len(sr10.dropna().to_numpy()) == (401 - 5)

    sr11 = sr9.replace([22, 323, 27, 0], -1)
    assert sr11.null_count == 1
    assert len(sr11.dropna().to_numpy()) == (401 - 1)

    # large input not containing nulls
    sr9 = sr9.fillna(-11)
    sr12 = sr9.replace([22, 323, 27, 0], None)
    assert sr12.null_count == 4
    assert len(sr12.dropna().to_numpy()) == (401 - 4)

    sr13 = sr9.replace([22, 323, 27, 0], -1)
    assert sr13.null_count == 0
    assert len(sr13.to_numpy()) == 401


def test_series_replace_with_nulls():
    a1 = np.array([0, 1, 2, 3, 4])

    # Numerical
    a2 = np.array([-10, 1, 2, 3, 4])
    sr1 = cudf.Series(a1)
    sr2 = sr1.replace(0, None).fillna(-10)
    assert_eq(a2, sr2.to_numpy())

    # List input
    a6 = np.array([-10, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [None, 6]).fillna(-10)
    assert_eq(a6, sr6.to_numpy())

    sr1 = cudf.Series([0, 1, 2, 3, 4, None])
    assert_eq(
        sr1.replace([0, 1], [5.5, 6.5]).fillna(-10),
        sr1.to_pandas().replace([0, 1], [5.5, 6.5]).fillna(-10),
    )

    # Series input
    a8 = np.array([-10, -10, -10, 3, 4, -10])
    sr8 = sr1.replace(cudf.Series([-10] * 3, index=sr1[:3]), None).fillna(-10)
    assert_eq(a8, sr8.to_numpy())

    a9 = np.array([-10, 6, 2, 3, 4, -10])
    sr9 = sr1.replace([0, 1], [None, 6]).fillna(-10)
    assert_eq(a9, sr9.to_numpy())


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([0, 1, None, 2, None], dtype=pd.Int8Dtype()),
        pd.Series([0, 1, np.nan, 2, np.nan]),
    ],
)
@pytest.mark.parametrize("fill_value", [10, pd.Series([10, 20, 30, 40, 50])])
def test_series_fillna_numerical(
    psr, numeric_types_as_str, fill_value, inplace
):
    if inplace:
        psr = psr.copy(deep=True)
    # TODO: These tests should use Pandas' nullable int type
    # when we support a recent enough version of Pandas
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
    if np.dtype(numeric_types_as_str).kind != "f" and psr.dtype.kind == "i":
        psr = psr.astype(
            cudf.utils.dtypes.np_dtypes_to_pandas_dtypes[
                np.dtype(numeric_types_as_str)
            ]
        )

    gsr = cudf.from_pandas(psr)

    if isinstance(fill_value, pd.Series):
        fill_value_cudf = cudf.from_pandas(fill_value)
    else:
        fill_value_cudf = fill_value

    expected = psr.fillna(fill_value, inplace=inplace)
    actual = gsr.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        expected = psr
        actual = gsr

    # TODO: Remove check_dtype when we have support
    # to compare with pandas nullable dtypes
    assert_eq(expected, actual, check_dtype=False)


def test_series_multiple_times_with_nulls():
    sr = cudf.Series([1, 2, 3, None])
    expected = cudf.Series([None, None, None, None], dtype=np.int64)

    for i in range(3):
        got = sr.replace([1, 2, 3], None)
        assert_eq(expected, got)
        # BUG: #2695
        # The following series will acquire a chunk of memory and update with
        # values, but these values may still linger even after the memory
        # gets released. This memory space might get used for replace in
        # subsequent calls and the memory used for mask may have junk values.
        # So, if it is not updated properly, the result would be wrong.
        # So, this will help verify that scenario.
        cudf.Series([1, 1, 1, None])


@pytest.mark.parametrize(
    "replacement", [128, 128.0, 128.5, 32769, 32769.0, 32769.5]
)
def test_numeric_series_replace_dtype(
    request, numeric_types_as_str, replacement
):
    request.applymarker(
        pytest.mark.xfail(
            condition=PANDAS_GT_214
            and (
                (
                    numeric_types_as_str == "int8"
                    and replacement in {128, 128.0, 32769, 32769.0}
                )
                or (
                    numeric_types_as_str == "int16"
                    and replacement in {32769, 32769.0}
                )
            ),
            reason="Pandas throws an AssertionError for these "
            "cases and asks us to log a bug, they are trying to "
            "avoid a RecursionError which cudf will not run into",
        )
    )
    psr = pd.Series([0, 1, 2, 3, 4, 5], dtype=numeric_types_as_str)
    sr = cudf.from_pandas(psr)

    expect = psr.replace(1, replacement)
    got = sr.replace(1, replacement)

    assert_eq(expect, got)

    # to_replace is a list, replacement is a scalar
    expect = psr.replace([2, 3], replacement)
    got = sr.replace([2, 3], replacement)

    assert_eq(expect, got)

    # If to_replace is a scalar and replacement is a list
    with pytest.raises(TypeError):
        sr.replace(0, [replacement, 2])

    # Both list of unequal length
    with pytest.raises(ValueError):
        sr.replace([0, 1], [replacement])

    # Both lists of equal length
    expect = psr.replace([2, 3], [replacement, replacement])
    got = sr.replace([2, 3], [replacement, replacement])
    assert_eq(expect, got)
