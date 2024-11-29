# Copyright (c) 2020-2024, NVIDIA CORPORATION.
import datetime
import decimal
import hashlib
import operator
import re
from collections import OrderedDict, defaultdict
from string import ascii_letters, digits

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.errors import MixedTypeError
from cudf.testing import assert_eq
from cudf.testing._utils import (
    NUMERIC_TYPES,
    SERIES_OR_INDEX_NAMES,
    TIMEDELTA_TYPES,
    assert_exceptions_equal,
    expect_warning_if,
    gen_rand,
)


def _series_na_data():
    return [
        pd.Series([0, 1, 2, np.nan, 4, None, 6]),
        pd.Series(
            [0, 1, 2, np.nan, 4, None, 6],
            index=["q", "w", "e", "r", "t", "y", "u"],
            name="a",
        ),
        pd.Series([0, 1, 2, 3, 4]),
        pd.Series(["a", "b", "u", "h", "d"]),
        pd.Series([None, None, np.nan, None, np.inf, -np.inf]),
        pd.Series([], dtype="float64"),
        pd.Series(
            [pd.NaT, pd.Timestamp("1939-05-27"), pd.Timestamp("1940-04-25")]
        ),
        pd.Series([np.nan]),
        pd.Series([None]),
        pd.Series(["a", "b", "", "c", None, "e"]),
    ]


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 24, "d": 1010},
        {"a": 1},
        {1: "a", 2: "b", 24: "c", 1010: "d"},
        {1: "a"},
    ],
)
def test_series_init_dict(data):
    pandas_series = pd.Series(data)
    cudf_series = cudf.Series(data)

    assert_eq(pandas_series, cudf_series)


@pytest.mark.parametrize(
    "data",
    [
        {
            "a": [1, 2, 3],
            "b": [2, 3, 5],
            "c": [24, 12212, 22233],
            "d": [1010, 101010, 1111],
        },
        {"a": [1]},
    ],
)
def test_series_init_dict_lists(data):
    assert_eq(pd.Series(data), cudf.Series(data))


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4],
        [1.0, 12.221, 12.34, 13.324, 324.3242],
        [-10, -1111, 100, 11, 133],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [10, 11, 12, 13],
        [0.1, 0.002, 324.2332, 0.2342],
        [-10, -1111, 100, 11, 133],
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_concat_basic(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = pd.concat([psr, other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, other_gs], ignore_index=ignore_index)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [
            "abc",
            "def",
            "this is a string",
            "this is another string",
            "a",
            "b",
            "c",
        ],
        ["a"],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [
            "abc",
            "def",
            "this is a string",
            "this is another string",
            "a",
            "b",
            "c",
        ],
        ["a"],
        ["1", "2", "3", "4", "5"],
        ["+", "-", "!", "_", "="],
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_concat_basic_str(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = pd.concat([psr, other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, other_gs], ignore_index=ignore_index)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series(
            [
                "abc",
                "def",
                "this is a string",
                "this is another string",
                "a",
                "b",
                "c",
            ],
            index=[10, 20, 30, 40, 50, 60, 70],
        ),
        pd.Series(["a"], index=[2]),
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        pd.Series(
            [
                "abc",
                "def",
                "this is a   string",
                "this is another string",
                "a",
                "b",
                "c",
            ],
            index=[10, 20, 30, 40, 50, 60, 70],
        ),
        pd.Series(["a"], index=[133]),
        pd.Series(["1", "2", "3", "4", "5"], index=[-10, 22, 33, 44, 49]),
        pd.Series(["+", "-", "!", "_", "="], index=[11, 22, 33, 44, 2]),
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_concat_series_with_index(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = cudf.from_pandas(others)

    expected = pd.concat([psr, other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, other_gs], ignore_index=ignore_index)

    assert_eq(expected, actual)


def test_series_concat_error_mixed_types():
    gsr = cudf.Series([1, 2, 3, 4])
    other = cudf.Series(["a", "b", "c", "d"])

    with pytest.raises(
        TypeError,
        match="cudf does not support mixed types, please type-cast "
        "both series to same dtypes.",
    ):
        cudf.concat([gsr, other])

    with pytest.raises(
        TypeError,
        match="cudf does not support mixed types, please type-cast "
        "both series to same dtypes.",
    ):
        cudf.concat([gsr, gsr, other, gsr, other])


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"]),
        pd.Series(
            [1.0, 12.221, 12.34, 13.324, 324.3242],
            index=[
                "float one",
                "float two",
                "float three",
                "float four",
                "float five",
            ],
        ),
        pd.Series(
            [-10, -1111, 100, 11, 133],
            index=["one", "two", "three", "four", "five"],
        ),
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [
            pd.Series([10, 11, 12, 13], index=["a", "b", "c", "d"]),
            pd.Series([12, 14, 15, 27], index=["d", "e", "z", "x"]),
        ],
        [
            pd.Series([10, 11, 12, 13], index=["a", "b", "c", "d"]),
            pd.Series([12, 14, 15, 27], index=["d", "e", "z", "x"]),
        ]
        * 25,
        [
            pd.Series(
                [0.1, 0.002, 324.2332, 0.2342], index=["-", "+", "%", "#"]
            ),
            pd.Series([12, 14, 15, 27], index=["d", "e", "z", "x"]),
        ]
        * 46,
        [
            pd.Series(
                [-10, -1111, 100, 11, 133],
                index=["aa", "vv", "bb", "dd", "ll"],
            )
        ],
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_concat_list_series_with_index(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = [cudf.from_pandas(obj) for obj in others]

    expected = pd.concat([psr, *other_ps], ignore_index=ignore_index)
    actual = cudf.concat([gsr, *other_gs], ignore_index=ignore_index)

    assert_eq(expected, actual)


def test_series_concat_existing_buffers():
    a1 = np.arange(10, dtype=np.float64)
    gs = cudf.Series(a1)

    # Add new buffer
    a2 = cudf.Series(np.arange(5))
    gs = cudf.concat([gs, a2])
    assert len(gs) == 15
    np.testing.assert_equal(gs.to_numpy(), np.hstack([a1, a2.to_numpy()]))

    # Ensure appending to previous buffer
    a3 = cudf.Series(np.arange(3))
    gs = cudf.concat([gs, a3])
    assert len(gs) == 18
    a4 = np.hstack([a1, a2.to_numpy(), a3.to_numpy()])
    np.testing.assert_equal(gs.to_numpy(), a4)

    # Appending different dtype
    a5 = cudf.Series(np.array([1, 2, 3], dtype=np.int32))
    a6 = cudf.Series(np.array([4.5, 5.5, 6.5], dtype=np.float64))
    gs = cudf.concat([a5, a6])
    np.testing.assert_equal(
        gs.to_numpy(), np.hstack([a5.to_numpy(), a6.to_numpy()])
    )
    gs = cudf.concat([cudf.Series(a6), a5])
    np.testing.assert_equal(
        gs.to_numpy(), np.hstack([a6.to_numpy(), a5.to_numpy()])
    )


def test_series_column_iter_error():
    gs = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(gs)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.items()

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.iteritems()

    with pytest.raises(TypeError):
        iter(gs._column)


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, None, 4.0, 5.0],
        ["a", "b", "c", "d", "e"],
        ["a", "b", None, "d", "e"],
        [None, None, None, None, None],
        np.array(["1991-11-20", "2004-12-04"], dtype=np.datetime64),
        np.array(["1991-11-20", None], dtype=np.datetime64),
        np.array(
            ["1991-11-20 05:15:00", "2004-12-04 10:00:00"], dtype=np.datetime64
        ),
        np.array(["1991-11-20 05:15:00", None], dtype=np.datetime64),
    ],
)
def test_series_tolist(data):
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)

    with pytest.raises(
        TypeError,
        match=re.escape(
            r"cuDF does not support conversion to host memory "
            r"via the `tolist()` method. Consider using "
            r"`.to_arrow().to_pylist()` to construct a Python list."
        ),
    ):
        gsr.tolist()


@pytest.mark.parametrize(
    "data",
    [[], [None, None], ["a"], ["a", "b", "c"] * 500, [1.0, 2.0, 0.3] * 57],
)
def test_series_size(data):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    assert_eq(psr.size, gsr.size)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_series_describe_numeric(dtype):
    ps = pd.Series([0, 1, 2, 3, 1, 2, 3], dtype=dtype)
    gs = cudf.from_pandas(ps)
    actual = gs.describe()
    expected = ps.describe()

    assert_eq(expected, actual, check_dtype=True)


@pytest.mark.parametrize("dtype", ["datetime64[ns]"])
def test_series_describe_datetime(dtype):
    # Note that other datetime units are not tested because pandas does not
    # support them. When specified coarser units, cuDF datetime columns cannot
    # represent fractional time for quantiles of the column, which may require
    # interpolation, this differs from pandas which always stay in [ns] unit.
    gs = cudf.Series([0, 1, 2, 3, 1, 2, 3], dtype=dtype)
    ps = gs.to_pandas()

    # Treating datetimes as categoricals is deprecated in pandas and will
    # be removed in future. Future behavior is treating datetime as numeric.
    expected = ps.describe()
    actual = gs.describe()

    assert_eq(expected.astype("str"), actual)


@pytest.mark.parametrize("dtype", TIMEDELTA_TYPES)
def test_series_describe_timedelta(dtype):
    ps = pd.Series([0, 1, 2, 3, 1, 2, 3], dtype=dtype)
    gs = cudf.from_pandas(ps)

    expected = ps.describe()
    actual = gs.describe()

    assert_eq(actual, expected.astype("str"))


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(["a", "b", "c", "d", "e", "a"]),
        pd.Series([True, False, True, True, False]),
        pd.Series([], dtype="str"),
        pd.Series(["a", "b", "c", "a"], dtype="category"),
        pd.Series(["d", "e", "f"], dtype="category"),
        pd.Series(pd.Categorical(["d", "e", "f"], categories=["f", "e", "d"])),
        pd.Series(
            pd.Categorical(
                ["d", "e", "f"], categories=["f", "e", "d"], ordered=True
            )
        ),
    ],
)
def test_series_describe_other_types(ps):
    gs = cudf.from_pandas(ps)

    expected = ps.describe()
    actual = gs.describe()

    if len(ps) == 0:
        assert_eq(expected.fillna("a").astype("str"), actual.fillna("a"))
    else:
        assert_eq(expected.astype("str"), actual)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 2, 1],
        [1, 2, None, 3, 1, 1],
        [],
        ["a", "b", "c", None, "z", "a"],
    ],
)
@pytest.mark.parametrize("use_na_sentinel", [True, False])
def test_series_factorize_use_na_sentinel(data, use_na_sentinel):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas(nullable=True)

    expected_labels, expected_cats = psr.factorize(
        use_na_sentinel=use_na_sentinel, sort=True
    )
    actual_labels, actual_cats = gsr.factorize(
        use_na_sentinel=use_na_sentinel, sort=True
    )
    assert_eq(expected_labels, actual_labels.get())
    assert_eq(expected_cats, actual_cats.to_pandas(nullable=True))


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 2, 1],
        [1, 2, None, 3, 1, 1],
        [],
        ["a", "b", "c", None, "z", "a"],
    ],
)
@pytest.mark.parametrize("sort", [True, False])
def test_series_factorize_sort(data, sort):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas(nullable=True)

    expected_labels, expected_cats = psr.factorize(sort=sort)
    actual_labels, actual_cats = gsr.factorize(sort=sort)
    assert_eq(expected_labels, actual_labels.get())
    assert_eq(expected_cats, actual_cats.to_pandas(nullable=True))


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_series_datetime_value_counts(data, nulls, normalize, dropna):
    psr = data.copy()
    rng = np.random.default_rng(seed=0)
    if len(data) > 0:
        if nulls == "one":
            p = rng.integers(0, len(data))
            psr[p] = None
        elif nulls == "some":
            p = rng.integers(0, len(data), 2)
            psr[p] = None

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


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("num_elements", [10, 100, 1000])
def test_categorical_value_counts(dropna, normalize, num_elements):
    # create categorical series
    rng = np.random.default_rng(seed=12)
    pd_cat = pd.Categorical(
        pd.Series(
            rng.choice(list(ascii_letters + digits), num_elements),
            dtype="category",
        )
    )

    # gdf
    gdf = cudf.DataFrame()
    gdf["a"] = cudf.Series.from_categorical(pd_cat)
    gdf_value_counts = gdf["a"].value_counts(
        dropna=dropna, normalize=normalize
    )

    # pandas
    pdf = pd.DataFrame()
    pdf["a"] = pd_cat
    pdf_value_counts = pdf["a"].value_counts(
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


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_series_value_counts(dropna, normalize):
    rng = np.random.default_rng(seed=0)
    for size in [10**x for x in range(5)]:
        arr = rng.integers(low=-1, high=10, size=size)
        mask = arr != -1
        sr = cudf.Series.from_masked_array(
            arr, cudf.Series(mask)._column.as_mask()
        )
        sr.name = "col"

        expect = (
            sr.to_pandas()
            .value_counts(dropna=dropna, normalize=normalize)
            .sort_index()
        )
        got = sr.value_counts(dropna=dropna, normalize=normalize).sort_index()

        assert_eq(expect, got, check_dtype=True, check_index_type=False)


@pytest.mark.parametrize("bins", [1, 2, 3])
def test_series_value_counts_bins(bins):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(bins=bins)
    got = gsr.value_counts(bins=bins)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=True)


@pytest.mark.parametrize("bins", [1, 2, 3])
@pytest.mark.parametrize("dropna", [True, False])
def test_series_value_counts_bins_dropna(bins, dropna):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, np.nan])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(bins=bins, dropna=dropna)
    got = gsr.value_counts(bins=bins, dropna=dropna)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=True)


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
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


@pytest.mark.parametrize(
    "gs",
    [
        cudf.Series([1, 2, 3]),
        cudf.Series([None]),
        cudf.Series([4]),
        cudf.Series([2, 3, -1, 0, 1], name="test name"),
        cudf.Series(
            [1, 2, 3, None, 2, 1], index=["a", "v", "d", "e", "f", "g"]
        ),
        cudf.Series([1, 2, 3, None, 2, 1, None], name="abc"),
        cudf.Series(["ab", "bc", "ab", None, "bc", None, None]),
        cudf.Series([None, None, None, None, None], dtype="str"),
        cudf.Series([None, None, None, None, None]),
        cudf.Series(
            [
                123213,
                23123,
                123123,
                12213123,
                12213123,
                12213123,
                23123,
                2312323123,
                None,
                None,
            ],
            dtype="timedelta64[ns]",
        ),
        cudf.Series(
            [
                None,
                1,
                2,
                3242434,
                3233243,
                1,
                2,
                1023,
                None,
                12213123,
                None,
                2312323123,
                None,
                None,
            ],
            dtype="datetime64[ns]",
        ),
        cudf.Series(name="empty series", dtype="float64"),
        cudf.Series(["a", "b", "c", " ", "a", "b", "z"], dtype="category"),
    ],
)
@pytest.mark.parametrize("dropna", [True, False])
def test_series_mode(gs, dropna):
    ps = gs.to_pandas()

    expected = ps.mode(dropna=dropna)
    actual = gs.mode(dropna=dropna)

    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.parametrize(
    "arr",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.repeat([-0.6459412758761901], 100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        np.arange(-100.5, 101.5, 1),
    ],
)
@pytest.mark.parametrize("decimals", [-5, -3, -1, 0, 1, 4, 12, np.int8(1)])
def test_series_round(arr, decimals):
    pser = pd.Series(arr)
    ser = cudf.Series(arr)
    result = ser.round(decimals)
    expected = pser.round(decimals)

    assert_eq(result, expected)
    rng = np.random.default_rng(seed=0)
    # with nulls, maintaining existing null mask
    arr = arr.astype("float64")  # for pandas nulls
    arr.ravel()[rng.choice(arr.shape[0], arr.shape[0] // 2, replace=False)] = (
        np.nan
    )

    pser = pd.Series(arr)
    ser = cudf.Series(arr)
    result = ser.round(decimals)
    expected = pser.round(decimals)

    assert_eq(result, expected)


def test_series_round_half_up():
    s = cudf.Series([0.0, 1.0, 1.2, 1.7, 0.5, 1.5, 2.5, None])
    expect = cudf.Series([0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0, None])
    got = s.round(how="half_up")
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "series",
    [
        cudf.Series([1.0, None, np.nan, 4.0], nan_as_null=False),
        cudf.Series([1.24430, None, np.nan, 4.423530], nan_as_null=False),
        cudf.Series([1.24430, np.nan, 4.423530], nan_as_null=False),
        cudf.Series([-1.24430, np.nan, -4.423530], nan_as_null=False),
        cudf.Series(np.repeat(np.nan, 100)),
    ],
)
@pytest.mark.parametrize("decimal", [0, 1, 2, 3])
def test_round_nan_as_null_false(series, decimal):
    pser = series.to_pandas()
    result = series.round(decimal)
    expected = pser.round(decimal)
    assert_eq(result, expected, atol=1e-10)


@pytest.mark.parametrize("ps", _series_na_data())
@pytest.mark.parametrize("nan_as_null", [True, False, None])
def test_series_isnull_isna(ps, nan_as_null):
    nan_contains = ps.apply(lambda x: isinstance(x, float) and np.isnan(x))
    if nan_as_null is False and (
        nan_contains.any() and not nan_contains.all() and ps.dtype == object
    ):
        with pytest.raises(MixedTypeError):
            cudf.Series.from_pandas(ps, nan_as_null=nan_as_null)
    else:
        gs = cudf.Series.from_pandas(ps, nan_as_null=nan_as_null)

        assert_eq(ps.isnull(), gs.isnull())
        assert_eq(ps.isna(), gs.isna())


@pytest.mark.parametrize("ps", _series_na_data())
@pytest.mark.parametrize("nan_as_null", [True, False, None])
def test_series_notnull_notna(ps, nan_as_null):
    nan_contains = ps.apply(lambda x: isinstance(x, float) and np.isnan(x))
    if nan_as_null is False and (
        nan_contains.any() and not nan_contains.all() and ps.dtype == object
    ):
        with pytest.raises(MixedTypeError):
            cudf.Series.from_pandas(ps, nan_as_null=nan_as_null)
    else:
        gs = cudf.Series.from_pandas(ps, nan_as_null=nan_as_null)

        assert_eq(ps.notnull(), gs.notnull())
        assert_eq(ps.notna(), gs.notna())


@pytest.mark.parametrize(
    "sr1", [pd.Series([10, 11, 12], index=["a", "b", "z"]), pd.Series(["a"])]
)
@pytest.mark.parametrize(
    "sr2",
    [pd.Series([], dtype="float64"), pd.Series(["a", "a", "c", "z", "A"])],
)
@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ],
)
def test_series_error_equality(sr1, sr2, op):
    gsr1 = cudf.from_pandas(sr1)
    gsr2 = cudf.from_pandas(sr2)

    assert_exceptions_equal(op, op, ([sr1, sr2],), ([gsr1, gsr2],))


def test_series_memory_usage():
    sr = cudf.Series([1, 2, 3, 4], dtype="int64")
    assert sr.memory_usage() == 32

    sliced_sr = sr[2:]
    assert sliced_sr.memory_usage() == 16

    sliced_sr[3] = None
    assert sliced_sr.memory_usage() == 80

    sr = cudf.Series(["hello world", "rapids ai", "abc", "z"])
    assert sr.memory_usage() == 44

    assert sr[3:].memory_usage() == 9  # z
    assert sr[:1].memory_usage() == 19  # hello world


@pytest.mark.parametrize(
    "sr,expected_psr",
    [
        (
            cudf.Series([1, 2, None, 3], dtype="uint8"),
            pd.Series([1, 2, None, 3], dtype=pd.UInt8Dtype()),
        ),
        (
            cudf.Series([23, None, None, 32], dtype="uint16"),
            pd.Series([23, None, None, 32], dtype=pd.UInt16Dtype()),
        ),
        (
            cudf.Series([None, 123, None, 1], dtype="uint32"),
            pd.Series([None, 123, None, 1], dtype=pd.UInt32Dtype()),
        ),
        (
            cudf.Series([234, 2323, 23432, None, None, 224], dtype="uint64"),
            pd.Series(
                [234, 2323, 23432, None, None, 224], dtype=pd.UInt64Dtype()
            ),
        ),
        (
            cudf.Series([-10, 1, None, -1, None, 3], dtype="int8"),
            pd.Series([-10, 1, None, -1, None, 3], dtype=pd.Int8Dtype()),
        ),
        (
            cudf.Series([111, None, 222, None, 13], dtype="int16"),
            pd.Series([111, None, 222, None, 13], dtype=pd.Int16Dtype()),
        ),
        (
            cudf.Series([11, None, 22, 33, None, 2, None, 3], dtype="int32"),
            pd.Series(
                [11, None, 22, 33, None, 2, None, 3], dtype=pd.Int32Dtype()
            ),
        ),
        (
            cudf.Series(
                [32431, None, None, 32322, 0, 10, -32324, None], dtype="int64"
            ),
            pd.Series(
                [32431, None, None, 32322, 0, 10, -32324, None],
                dtype=pd.Int64Dtype(),
            ),
        ),
        (
            cudf.Series(
                [True, None, False, None, False, True, True, False],
                dtype="bool_",
            ),
            pd.Series(
                [True, None, False, None, False, True, True, False],
                dtype=pd.BooleanDtype(),
            ),
        ),
        (
            cudf.Series(
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
                dtype="object",
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
            cudf.Series(
                [1, 2, None, 10.2, None],
                dtype="float32",
            ),
            pd.Series(
                [1, 2, None, 10.2, None],
                dtype=pd.Float32Dtype(),
            ),
        ),
    ],
)
def test_series_to_pandas_nullable_dtypes(sr, expected_psr):
    actual_psr = sr.to_pandas(nullable=True)

    assert_eq(actual_psr, expected_psr)


def test_series_pipe():
    psr = pd.Series([10, 20, 30, 40])
    gsr = cudf.Series([10, 20, 30, 40])

    def custom_add_func(sr, val):
        new_sr = sr + val
        return new_sr

    def custom_to_str_func(sr, val):
        new_sr = sr.astype("str") + val
        return new_sr

    expected = (
        psr.pipe(custom_add_func, 11)
        .pipe(custom_add_func, val=12)
        .pipe(custom_to_str_func, "rapids")
    )
    actual = (
        gsr.pipe(custom_add_func, 11)
        .pipe(custom_add_func, val=12)
        .pipe(custom_to_str_func, "rapids")
    )

    assert_eq(expected, actual)

    expected = (
        psr.pipe((custom_add_func, "sr"), val=11)
        .pipe(custom_add_func, val=1)
        .pipe(custom_to_str_func, "rapids-ai")
    )
    actual = (
        gsr.pipe((custom_add_func, "sr"), val=11)
        .pipe(custom_add_func, val=1)
        .pipe(custom_to_str_func, "rapids-ai")
    )

    assert_eq(expected, actual)


def test_series_pipe_error():
    psr = pd.Series([10, 20, 30, 40])
    gsr = cudf.Series([10, 20, 30, 40])

    def custom_add_func(sr, val):
        new_sr = sr + val
        return new_sr

    assert_exceptions_equal(
        lfunc=psr.pipe,
        rfunc=gsr.pipe,
        lfunc_args_and_kwargs=([(custom_add_func, "val")], {"val": 11}),
        rfunc_args_and_kwargs=([(custom_add_func, "val")], {"val": 11}),
    )


@pytest.mark.parametrize(
    "data",
    [cudf.Series([1, 2, 3]), cudf.Series([10, 11, 12], index=[1, 2, 3])],
)
@pytest.mark.parametrize(
    "other",
    [
        cudf.Series([4, 5, 6]),
        cudf.Series([4, 5, 6, 7, 8]),
        cudf.Series([4, np.nan, 6], nan_as_null=False),
        [4, np.nan, 6],
        {1: 9},
    ],
)
def test_series_update(data, other):
    gs = data.copy(deep=True)
    if isinstance(other, cudf.Series):
        g_other = other.copy(deep=True)
        p_other = g_other.to_pandas()
    else:
        g_other = other
        p_other = other

    ps = gs.to_pandas()

    ps.update(p_other)
    with expect_warning_if(
        isinstance(other, cudf.Series) and other.isna().any(), UserWarning
    ):
        gs.update(g_other)
    assert_eq(gs, ps)


@pytest.mark.parametrize(
    "data",
    [
        [1, None, 11, 2.0, np.nan],
        [np.nan],
        [None, None, None],
        [np.nan, 1, 10, 393.32, np.nan],
    ],
)
@pytest.mark.parametrize("nan_as_null", [True, False])
@pytest.mark.parametrize("fill_value", [1.2, 332, np.nan])
def test_fillna_with_nan(data, nan_as_null, fill_value):
    gs = cudf.Series(data, dtype="float64", nan_as_null=nan_as_null)
    ps = gs.to_pandas()

    expected = ps.fillna(fill_value)
    actual = gs.fillna(fill_value)

    assert_eq(expected, actual)


def test_fillna_categorical_with_non_categorical_raises():
    ser = cudf.Series([1, None], dtype="category")
    with pytest.raises(TypeError):
        ser.fillna(cudf.Series([1, 2]))


def test_fillna_categorical_with_different_categories_raises():
    ser = cudf.Series([1, None], dtype="category")
    with pytest.raises(TypeError):
        ser.fillna(cudf.Series([1, 2]), dtype="category")


def test_series_mask_mixed_dtypes_error():
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


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(["a"] * 20, index=range(0, 20)),
        pd.Series(["b", None] * 10, index=range(0, 20), name="ASeries"),
        pd.Series(
            ["b", None] * 5,
            index=pd.Index(list(range(10)), dtype="uint64"),
            name="BSeries",
        ),
    ],
)
@pytest.mark.parametrize(
    "labels",
    [
        [1],
        [0],
        1,
        5,
        [5, 9],
        pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        pd.Index([0, 1, 2, 3, 4], dtype="float32"),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_series_drop_labels(ps, labels, inplace):
    ps = ps.copy()
    gs = cudf.from_pandas(ps)

    expected = ps.drop(labels=labels, axis=0, inplace=inplace)
    actual = gs.drop(labels=labels, axis=0, inplace=inplace)

    if inplace:
        expected = ps
        actual = gs

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(["a"] * 20, index=range(0, 20)),
        pd.Series(["b", None] * 10, index=range(0, 20), name="ASeries"),
    ],
)
@pytest.mark.parametrize(
    "index",
    [[1], [0], 1, 5, [5, 9], pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_series_drop_index(ps, index, inplace):
    ps = ps.copy()
    gs = cudf.from_pandas(ps)

    expected = ps.drop(index=index, inplace=inplace)
    actual = gs.drop(index=index, inplace=inplace)

    if inplace:
        expected = ps
        actual = gs

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(
            ["a" if i % 2 == 0 else "b" for i in range(0, 10)],
            index=pd.MultiIndex(
                levels=[
                    ["lama", "cow", "falcon"],
                    ["speed", "weight", "length"],
                ],
                codes=[
                    [0, 0, 0, 1, 1, 1, 2, 2, 2, 1],
                    [0, 1, 2, 0, 1, 2, 0, 1, 2, 1],
                ],
            ),
            name="abc",
        )
    ],
)
@pytest.mark.parametrize(
    "index,level",
    [
        ("cow", 0),
        ("lama", 0),
        ("falcon", 0),
        ("speed", 1),
        ("weight", 1),
        ("length", 1),
        (
            "cow",
            None,
        ),
        (
            "lama",
            None,
        ),
        (
            "falcon",
            None,
        ),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_series_drop_multiindex(ps, index, level, inplace):
    ps = ps.copy()
    gs = cudf.from_pandas(ps)

    expected = ps.drop(index=index, inplace=inplace, level=level)
    actual = gs.drop(index=index, inplace=inplace, level=level)

    if inplace:
        expected = ps
        actual = gs

    assert_eq(expected, actual)


def test_series_drop_edge_inputs():
    gs = cudf.Series([42], name="a")
    ps = gs.to_pandas()

    assert_eq(ps.drop(columns=["b"]), gs.drop(columns=["b"]))

    assert_eq(ps.drop(columns="b"), gs.drop(columns="b"))

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=(["a"], {"columns": "a", "axis": 1}),
        rfunc_args_and_kwargs=(["a"], {"columns": "a", "axis": 1}),
    )

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=([], {}),
        rfunc_args_and_kwargs=([], {}),
    )

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=(["b"], {"axis": 1}),
        rfunc_args_and_kwargs=(["b"], {"axis": 1}),
    )


def test_series_drop_raises():
    gs = cudf.Series([10, 20, 30], index=["x", "y", "z"], name="c")
    ps = gs.to_pandas()

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=(["p"],),
        rfunc_args_and_kwargs=(["p"],),
    )

    # dtype specified mismatch
    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=([3],),
        rfunc_args_and_kwargs=([3],),
    )

    expect = ps.drop("p", errors="ignore")
    actual = gs.drop("p", errors="ignore")

    assert_eq(actual, expect)


@pytest.mark.parametrize(
    "data",
    [[[1, 2, 3], None, [4], [], [5, 6]], [1, 2, 3, 4, 5]],
)
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize(
    "p_index",
    [
        None,
        ["ia", "ib", "ic", "id", "ie"],
        pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b")]
        ),
    ],
)
def test_explode(data, ignore_index, p_index):
    pdf = pd.Series(data, index=p_index, name="someseries")
    gdf = cudf.from_pandas(pdf)

    expect = pdf.explode(ignore_index)
    got = gdf.explode(ignore_index)

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            [cudf.Series([1, 2, 3]), cudf.Series([10, 20])],
            cudf.Series([[1, 2, 3], [10, 20]]),
        ),
        (
            [cudf.Series([1, 2, 3]), None, cudf.Series([10, 20, np.nan])],
            cudf.Series([[1, 2, 3], None, [10, 20, np.nan]]),
        ),
        (
            [cp.array([5, 6]), cudf.NA, cp.array([1])],
            cudf.Series([[5, 6], None, [1]]),
        ),
        (
            [None, None, None, None, None, cudf.Series([10, 20])],
            cudf.Series([None, None, None, None, None, [10, 20]]),
        ),
    ],
)
def test_nested_series_from_sequence_data(data, expected):
    actual = cudf.Series(data)
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        cp.ones(5, dtype=cp.float16),
        np.ones(5, dtype="float16"),
        pd.Series([0.1, 1.2, 3.3], dtype="float16"),
        pytest.param(
            pa.array(np.ones(5, dtype="float16")),
            marks=pytest.mark.xfail(
                reason="https://issues.apache.org/jira/browse/ARROW-13762"
            ),
        ),
    ],
)
def test_series_raises_float16(data):
    with pytest.raises(TypeError):
        cudf.Series(data)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(0, 3, 1),
        [3.0, 1.0, np.nan],
        ["a", "z", None],
        pd.RangeIndex(4, -1, -2),
    ],
)
@pytest.mark.parametrize("axis", [0, "index"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_series_sort_index(
    index, axis, ascending, inplace, ignore_index, na_position
):
    ps = pd.Series([10, 3, 12], index=index)
    gs = cudf.from_pandas(ps)

    expected = ps.sort_index(
        axis=axis,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )
    got = gs.sort_index(
        axis=axis,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )

    if inplace is True:
        assert_eq(ps, gs, check_index_type=True)
    else:
        assert_eq(expected, got, check_index_type=True)


@pytest.mark.parametrize(
    "method", ["md5", "sha1", "sha224", "sha256", "sha384", "sha512"]
)
def test_series_hash_values(method):
    inputs = cudf.Series(
        [
            "",
            "0",
            "A 56 character string to test message padding algorithm.",
            "A 63 character string to test message padding algorithm, again.",
            "A 64 character string to test message padding algorithm, again!!",
            (
                "A very long (greater than 128 bytes/char string) to execute "
                "a multi hash-step data point in the hash function being "
                "tested. This string needed to be longer."
            ),
            "All work and no play makes Jack a dull boy",
            "!\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~",
            "\x00\x00\x00\x10\x00\x00\x00\x00",
            "\x00\x00\x00\x00",
        ]
    )

    def hashlib_compute_digest(data):
        hasher = getattr(hashlib, method)()
        hasher.update(data.encode("utf-8"))
        return hasher.hexdigest()

    hashlib_validation = inputs.to_pandas().apply(hashlib_compute_digest)
    validation_results = cudf.Series(hashlib_validation)
    hash_values = inputs.hash_values(method=method)
    assert_eq(hash_values, validation_results)


def test_series_hash_values_invalid_method():
    inputs = cudf.Series(["", "0"])
    with pytest.raises(ValueError):
        inputs.hash_values(method="invalid_method")


def test_set_index_unequal_length():
    s = cudf.Series(dtype="float64")
    with pytest.raises(ValueError):
        s.index = [1, 2, 3]


@pytest.mark.parametrize(
    "lhs, rhs", [("a", "a"), ("a", "b"), (1, 1.0), (None, None), (None, "a")]
)
def test_equals_names(lhs, rhs):
    lhs = cudf.Series([1, 2], name=lhs)
    rhs = cudf.Series([1, 2], name=rhs)

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data", [[True, False, None, True, False], [None, None], []]
)
@pytest.mark.parametrize("bool_dtype", ["bool", "boolean", pd.BooleanDtype()])
def test_nullable_bool_dtype_series(data, bool_dtype):
    psr = pd.Series(data, dtype=pd.BooleanDtype())
    gsr = cudf.Series(data, dtype=bool_dtype)

    assert_eq(psr, gsr.to_pandas(nullable=True))


@pytest.mark.parametrize("level", [None, 0, "l0", 1, ["l0", 1]])
@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("original_name", [None, "original_ser"])
@pytest.mark.parametrize("name", [None, "ser", no_default])
@pytest.mark.parametrize("inplace", [True, False])
def test_reset_index(level, drop, inplace, original_name, name):
    midx = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["l0", None]
    )
    ps = pd.Series(range(4), index=midx, name=original_name)
    gs = cudf.from_pandas(ps)

    if not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )

    expect = ps.reset_index(level=level, drop=drop, name=name, inplace=inplace)

    got = gs.reset_index(level=level, drop=drop, name=name, inplace=inplace)
    if inplace:
        expect = ps
        got = gs

    assert_eq(expect, got)


@pytest.mark.parametrize("level", [None, 0, 1, [None]])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("original_name", [None, "original_ser"])
@pytest.mark.parametrize("name", [None, "ser"])
def test_reset_index_dup_level_name(level, drop, inplace, original_name, name):
    # midx levels are named [None, None]
    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    ps = pd.Series(range(4), index=midx, name=original_name)
    gs = cudf.from_pandas(ps)
    if level == [None] or not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )

    expect = ps.reset_index(level=level, drop=drop, inplace=inplace, name=name)
    got = gs.reset_index(level=level, drop=drop, inplace=inplace, name=name)
    if inplace:
        expect = ps
        got = gs

    assert_eq(expect, got)


@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("original_name", [None, "original_ser"])
@pytest.mark.parametrize("name", [None, "ser"])
def test_reset_index_named(drop, inplace, original_name, name):
    ps = pd.Series(range(4), index=["x", "y", "z", "w"], name=original_name)
    gs = cudf.from_pandas(ps)

    ps.index.name = "cudf"
    gs.index.name = "cudf"

    if not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )

    expect = ps.reset_index(drop=drop, inplace=inplace, name=name)
    got = gs.reset_index(drop=drop, inplace=inplace, name=name)

    if inplace:
        expect = ps
        got = gs

    assert_eq(expect, got)


def test_reset_index_dup_level_name_exceptions():
    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    ps = pd.Series(range(4), index=midx)
    gs = cudf.from_pandas(ps)

    # Should specify duplicate level names with level number.
    assert_exceptions_equal(
        lfunc=ps.reset_index,
        rfunc=gs.reset_index,
        lfunc_args_and_kwargs=(
            [],
            {"level": [None]},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"level": [None]},
        ),
    )

    # Cannot use drop=False and inplace=True to turn a series into dataframe.
    assert_exceptions_equal(
        lfunc=ps.reset_index,
        rfunc=gs.reset_index,
        lfunc_args_and_kwargs=(
            [],
            {"drop": False, "inplace": True},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"drop": False, "inplace": True},
        ),
    )

    # Pandas raises the above exception should these two inputs crosses.
    assert_exceptions_equal(
        lfunc=ps.reset_index,
        rfunc=gs.reset_index,
        lfunc_args_and_kwargs=(
            [],
            {"level": [None], "drop": False, "inplace": True},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"level": [None], "drop": False, "inplace": True},
        ),
    )


def test_series_add_prefix():
    cd_s = cudf.Series([1, 2, 3, 4])
    pd_s = cd_s.to_pandas()

    got = cd_s.add_prefix("item_")
    expected = pd_s.add_prefix("item_")

    assert_eq(got, expected)


def test_series_add_suffix():
    cd_s = cudf.Series([1, 2, 3, 4])
    pd_s = cd_s.to_pandas()

    got = cd_s.add_suffix("_item")
    expected = pd_s.add_suffix("_item")

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "cudf_series",
    [
        cudf.Series([0.25, 0.5, 0.2, -0.05]),
        cudf.Series([0, 1, 2, np.nan, 4, cudf.NA, 6]),
    ],
)
@pytest.mark.parametrize("lag", [1, 2, 3, 4])
def test_autocorr(cudf_series, lag):
    psr = cudf_series.to_pandas()

    cudf_corr = cudf_series.autocorr(lag=lag)

    # autocorrelation is undefined (nan) for less than two entries, but pandas
    # short-circuits when there are 0 entries and bypasses the numpy function
    # call that generates an error.
    num_both_valid = (psr.notna() & psr.shift(lag).notna()).sum()
    with expect_warning_if(num_both_valid == 1, RuntimeWarning):
        pd_corr = psr.autocorr(lag=lag)

    assert_eq(pd_corr, cudf_corr)


@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3],
        ["abc", "a", None, "hello world", "foo buzz", "", None, "rapids ai"],
    ],
)
def test_series_transpose(data):
    psr = pd.Series(data=data)
    csr = cudf.Series(data=data)

    cudf_transposed = csr.transpose()
    pd_transposed = psr.transpose()
    cudf_property = csr.T
    pd_property = psr.T

    assert_eq(pd_transposed, cudf_transposed)
    assert_eq(pd_property, cudf_property)
    assert_eq(cudf_transposed, csr)


@pytest.mark.parametrize(
    "data",
    [1, 3, 5, 7, 7],
)
def test_series_nunique(data):
    cd_s = cudf.Series(data)
    pd_s = cd_s.to_pandas()

    actual = cd_s.nunique()
    expected = pd_s.nunique()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [1, 3, 5, 7, 7],
)
def test_series_nunique_index(data):
    cd_s = cudf.Series(data)
    pd_s = cd_s.to_pandas()

    actual = cd_s.index.nunique()
    expected = pd_s.index.nunique()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 4],
        ["a", "b", "c"],
        [1.2, 2.2, 4.5],
        [np.nan, np.nan],
        [None, None, None],
    ],
)
def test_axes(data):
    csr = cudf.Series(data)
    psr = csr.to_pandas()

    expected = psr.axes
    actual = csr.axes

    for e, a in zip(expected, actual):
        assert_eq(e, a)


def test_series_truncate():
    csr = cudf.Series([1, 2, 3, 4])
    psr = csr.to_pandas()

    assert_eq(csr.truncate(), psr.truncate())
    assert_eq(csr.truncate(1, 2), psr.truncate(1, 2))
    assert_eq(csr.truncate(before=1, after=2), psr.truncate(before=1, after=2))


def test_series_truncate_errors():
    csr = cudf.Series([1, 2, 3, 4])
    with pytest.raises(ValueError):
        csr.truncate(axis=1)
    with pytest.raises(ValueError):
        csr.truncate(copy=False)

    csr.index = [3, 2, 1, 6]
    psr = csr.to_pandas()
    assert_exceptions_equal(
        lfunc=csr.truncate,
        rfunc=psr.truncate,
    )


def test_series_truncate_datetimeindex():
    dates = cudf.date_range(
        "2021-01-01 23:45:00", "2021-01-02 23:46:00", freq="s"
    )
    csr = cudf.Series(range(len(dates)), index=dates)
    psr = csr.to_pandas()

    assert_eq(
        csr.truncate(
            before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"
        ),
        psr.truncate(
            before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"
        ),
    )


@pytest.mark.parametrize(
    "data",
    [
        [],
        [0, 12, 14],
        [0, 14, 12, 12, 3, 10, 12, 14],
        np.random.default_rng(seed=0).integers(-100, 100, 200),
        pd.Series([0.0, 1.0, None, 10.0]),
        [None, None, None, None],
        [np.nan, None, -1, 2, 3],
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        np.random.default_rng(seed=0).integers(-100, 100, 10),
        [],
        [np.nan, None, -1, 2, 3],
        [1.0, 12.0, None, None, 120],
        [0, 14, 12, 12, 3, 10, 12, 14, None],
        [None, None, None],
        ["0", "12", "14"],
        ["0", "12", "14", "a"],
    ],
)
def test_isin_numeric(data, values):
    rng = np.random.default_rng(seed=0)
    index = rng.integers(0, 100, len(data))
    psr = pd.Series(data, index=index)
    gsr = cudf.Series.from_pandas(psr, nan_as_null=False)

    expected = psr.isin(values)
    got = gsr.isin(values)

    assert_eq(got, expected)


@pytest.mark.xfail(raises=TypeError)
def test_fill_new_category():
    gs = cudf.Series(pd.Categorical(["a", "b", "c"]))
    gs[0:1] = "d"


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warning newly introduced in pandas-2.2.0",
)
@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(
            ["2018-01-01", "2019-04-03", None, "2019-12-30"],
            dtype="datetime64[ns]",
        ),
        pd.Series(
            [
                "2018-01-01",
                "2019-04-03",
                None,
                "2019-12-30",
                "2018-01-01",
                "2018-01-01",
            ],
            dtype="datetime64[ns]",
        ),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        [1514764800000000000, 1577664000000000000],
        [
            1514764800000000000,
            1577664000000000000,
            1577664000000000000,
            1577664000000000000,
            1514764800000000000,
        ],
        ["2019-04-03", "2019-12-30", "2012-01-01"],
        [
            "2012-01-01",
            "2012-01-01",
            "2012-01-01",
            "2019-04-03",
            "2019-12-30",
            "2012-01-01",
        ],
    ],
)
def test_isin_datetime(data, values):
    psr = pd.Series(data)
    gsr = cudf.Series.from_pandas(psr)

    is_len_str = isinstance(next(iter(values), None), str) and len(data)
    with expect_warning_if(is_len_str):
        got = gsr.isin(values)
    with expect_warning_if(is_len_str):
        expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(["this", "is", None, "a", "test"]),
        pd.Series(["test", "this", "test", "is", None, "test", "a", "test"]),
        pd.Series(["0", "12", "14"]),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["this", "is"],
        [None, None, None],
        ["12", "14", "19"],
        [12, 14, 19],
        ["is", "this", "is", "this", "is"],
    ],
)
def test_isin_string(data, values):
    psr = pd.Series(data)
    gsr = cudf.Series.from_pandas(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(["a", "b", "c", "c", "c", "d", "e"], dtype="category"),
        pd.Series(["a", "b", None, "c", "d", "e"], dtype="category"),
        pd.Series([0, 3, 10, 12], dtype="category"),
        pd.Series([0, 3, 10, 12, 0, 10, 3, 0, 0, 3, 3], dtype="category"),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["a", "b", None, "f", "words"],
        ["0", "12", None, "14"],
        [0, 10, 12, None, 39, 40, 1000],
        [0, 0, 0, 0, 3, 3, 3, None, 1, 2, 3],
    ],
)
def test_isin_categorical(data, values):
    psr = pd.Series(data)
    gsr = cudf.Series.from_pandas(psr)

    got = gsr.isin(values)
    expected = psr.isin(values)
    assert_eq(got, expected)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("period", [-1, -5, -10, -20, 0, 1, 5, 10, 20])
@pytest.mark.parametrize("data_empty", [False, True])
def test_diff(dtype, period, data_empty):
    if data_empty:
        data = None
    else:
        if dtype == np.int8:
            # to keep data in range
            data = gen_rand(dtype, 100000, low=-2, high=2)
        else:
            data = gen_rand(dtype, 100000)

    gs = cudf.Series(data, dtype=dtype)
    ps = pd.Series(data, dtype=dtype)

    expected_outcome = ps.diff(period)
    diffed_outcome = gs.diff(period).astype(expected_outcome.dtype)

    if data_empty:
        assert_eq(diffed_outcome, expected_outcome, check_index_type=False)
    else:
        assert_eq(diffed_outcome, expected_outcome)


@pytest.mark.parametrize(
    "data",
    [
        ["a", "b", "c", "d", "e"],
    ],
)
def test_diff_unsupported_dtypes(data):
    gs = cudf.Series(data)
    with pytest.raises(
        TypeError,
        match=r"unsupported operand type\(s\)",
    ):
        gs.diff()


@pytest.mark.parametrize(
    "data",
    [
        pd.date_range("2020-01-01", "2020-01-06", freq="D"),
        [True, True, True, False, True, True],
        [1.0, 2.0, 3.5, 4.0, 5.0, -1.7],
        [1, 2, 3, 3, 4, 5],
        [np.nan, None, None, np.nan, np.nan, None],
    ],
)
def test_diff_many_dtypes(data):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)
    assert_eq(ps.diff(), gs.diff())
    assert_eq(ps.diff(periods=2), gs.diff(periods=2))


@pytest.mark.parametrize("num_rows", [1, 100])
@pytest.mark.parametrize("num_bins", [1, 10])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("dtype", [*NUMERIC_TYPES, "bool"])
@pytest.mark.parametrize("series_bins", [True, False])
def test_series_digitize(num_rows, num_bins, right, dtype, series_bins):
    rng = np.random.default_rng(seed=0)
    data = rng.integers(0, 100, num_rows).astype(dtype)
    bins = np.unique(np.sort(rng.integers(2, 95, num_bins).astype(dtype)))
    s = cudf.Series(data)
    if series_bins:
        s_bins = cudf.Series(bins)
        indices = s.digitize(s_bins, right)
    else:
        indices = s.digitize(bins, right)
    np.testing.assert_array_equal(
        np.digitize(data, bins, right), indices.to_numpy()
    )


def test_series_digitize_invalid_bins():
    rng = np.random.default_rng(seed=0)
    s = cudf.Series(rng.integers(0, 30, 80), dtype="int32")
    bins = cudf.Series([2, None, None, 50, 90], dtype="int32")

    with pytest.raises(
        ValueError, match="`bins` cannot contain null entries."
    ):
        _ = s.digitize(bins)


@pytest.mark.parametrize(
    "data,left,right",
    [
        ([0, 1, 2, 3, 4, 5, 10], 0, 5),
        ([0, 1, 2, 3, 4, 5, 10], 10, 1),
        ([0, 1, 2, 3, 4, 5], [0, 10, 11] * 2, [1, 2, 5] * 2),
        (["a", "few", "set", "of", "strings", "xyz", "abc"], "banana", "few"),
        (["a", "few", "set", "of", "strings", "xyz", "abc"], "phone", "hello"),
        (
            ["a", "few", "set", "of", "strings", "xyz", "abc"],
            ["a", "hello", "rapids", "ai", "world", "chars", "strs"],
            ["yes", "no", "hi", "bye", "test", "pass", "fail"],
        ),
        ([0, 1, 2, np.nan, 4, np.nan, 10], 10, 1),
    ],
)
@pytest.mark.parametrize("inclusive", ["both", "neither", "left", "right"])
def test_series_between(data, left, right, inclusive):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps, nan_as_null=False)

    expected = ps.between(left, right, inclusive=inclusive)
    actual = gs.between(left, right, inclusive=inclusive)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data,left,right",
    [
        ([0, 1, 2, None, 4, 5, 10], 0, 5),
        ([0, 1, 2, 3, None, 5, 10], 10, 1),
        ([None, 1, 2, 3, 4, None], [0, 10, 11] * 2, [1, 2, 5] * 2),
        (
            ["a", "few", "set", None, "strings", "xyz", "abc"],
            ["a", "hello", "rapids", "ai", "world", "chars", "strs"],
            ["yes", "no", "hi", "bye", "test", "pass", "fail"],
        ),
    ],
)
@pytest.mark.parametrize("inclusive", ["both", "neither", "left", "right"])
def test_series_between_with_null(data, left, right, inclusive):
    gs = cudf.Series(data)
    ps = gs.to_pandas(nullable=True)

    expected = ps.between(left, right, inclusive=inclusive)
    actual = gs.between(left, right, inclusive=inclusive)

    assert_eq(expected, actual.to_pandas(nullable=True))


def test_default_construction():
    s = cudf.Series([np.int8(8), np.int16(128)])
    assert s.dtype == np.dtype("i2")


@pytest.mark.parametrize(
    "data", [[0, 1, 2, 3, 4], range(5), [np.int8(8), np.int16(128)]]
)
def test_default_integer_bitwidth_construction(default_integer_bitwidth, data):
    s = cudf.Series(data)
    assert s.dtype == np.dtype(f"i{default_integer_bitwidth//8}")


@pytest.mark.parametrize("data", [[1.5, 2.5, 4.5], [1000, 2000, 4000, 3.14]])
def test_default_float_bitwidth_construction(default_float_bitwidth, data):
    s = cudf.Series(data)
    assert s.dtype == np.dtype(f"f{default_float_bitwidth//8}")


def test_series_ordered_dedup():
    # part of https://github.com/rapidsai/cudf/issues/11486
    rng = np.random.default_rng(seed=0)
    sr = cudf.Series(rng.integers(0, 100, 1000))
    # pandas unique() preserves order
    expect = pd.Series(sr.to_pandas().unique())
    got = cudf.Series._from_column(sr._column.unique())
    assert_eq(expect.values, got.values)


@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize("bool_scalar", [True, False])
def test_set_bool_error(dtype, bool_scalar):
    sr = cudf.Series([1, 2, 3], dtype=dtype)
    psr = sr.to_pandas(nullable=True)

    assert_exceptions_equal(
        lfunc=sr.__setitem__,
        rfunc=psr.__setitem__,
        lfunc_args_and_kwargs=([bool_scalar],),
        rfunc_args_and_kwargs=([bool_scalar],),
    )


def test_int64_equality():
    s = cudf.Series(np.asarray([2**63 - 10, 2**63 - 100], dtype=np.int64))
    assert (s != np.int64(2**63 - 1)).all()
    assert (s != cudf.Scalar(2**63 - 1, dtype=np.int64)).all()


@pytest.mark.parametrize("into", [dict, OrderedDict, defaultdict(list)])
def test_series_to_dict(into):
    gs = cudf.Series(["ab", "de", "zx"], index=[10, 20, 100])
    ps = gs.to_pandas()

    actual = gs.to_dict(into=into)
    expected = ps.to_dict(into=into)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        pytest.param(
            [np.nan, 10, 15, 16],
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/49818"
            ),
        ),
        [np.nan, None, 10, 20],
        ["ab", "zx", "pq"],
        ["ab", "zx", None, "pq"],
        [],
    ],
)
def test_series_hasnans(data):
    gs = cudf.Series(data, nan_as_null=False)
    ps = gs.to_pandas(nullable=True)

    # Check type to avoid mixing Python bool and NumPy bool
    assert isinstance(gs.hasnans, bool)
    assert gs.hasnans == ps.hasnans


@pytest.mark.parametrize(
    "data,index",
    [
        ([1, 2, 3], [10, 11, 12]),
        ([1, 2, 3, 1, 1, 2, 3, 2], [10, 20, 23, 24, 25, 26, 27, 28]),
        ([1, None, 2, None, 3, None, 3, 1], [5, 6, 7, 8, 9, 10, 11, 12]),
        ([np.nan, 1.0, np.nan, 5.4, 5.4, 1.0], ["a", "b", "c", "d", "e", "f"]),
        (
            ["lama", "cow", "lama", None, "beetle", "lama", None, None],
            [1, 4, 10, 11, 2, 100, 200, 400],
        ),
    ],
)
@pytest.mark.parametrize("keep", ["first", "last", False])
@pytest.mark.parametrize("name", [None, "a"])
def test_series_duplicated(data, index, keep, name):
    gs = cudf.Series(data, index=index, name=name)
    ps = gs.to_pandas()

    assert_eq(gs.duplicated(keep=keep), ps.duplicated(keep=keep))


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4],
        [10, 20, None, None],
    ],
)
@pytest.mark.parametrize("copy", [True, False])
def test_series_copy(data, copy):
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)

    new_psr = pd.Series(psr, copy=copy)
    new_gsr = cudf.Series(gsr, copy=copy)

    new_psr.iloc[0] = 999
    new_gsr.iloc[0] = 999

    assert_eq(psr, gsr)
    assert_eq(new_psr, new_gsr)


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 24, "d": 1010},
        {"a": 1},
    ],
)
@pytest.mark.parametrize(
    "index", [None, ["b", "c"], ["d", "a", "c", "b"], ["a"]]
)
def test_series_init_dict_with_index(data, index):
    pandas_series = pd.Series(data, index=index)
    cudf_series = cudf.Series(data, index=index)

    assert_eq(pandas_series, cudf_series)


@pytest.mark.parametrize("data", ["abc", None, 1, 3.7])
@pytest.mark.parametrize(
    "index", [None, ["b", "c"], ["d", "a", "c", "b"], ["a"]]
)
def test_series_init_scalar_with_index(data, index):
    pandas_series = pd.Series(data, index=index)
    cudf_series = cudf.Series(data, index=index)

    assert_eq(
        pandas_series,
        cudf_series,
        check_index_type=data is not None or index is not None,
        check_dtype=data is not None,
    )


def test_series_init_error():
    assert_exceptions_equal(
        lfunc=pd.Series,
        rfunc=cudf.Series,
        lfunc_args_and_kwargs=([], {"data": [11], "index": [10, 11]}),
        rfunc_args_and_kwargs=([], {"data": [11], "index": [10, 11]}),
    )


def test_series_init_from_series_and_index():
    ser = cudf.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
    result = cudf.Series(ser, index=list("abcd"))
    expected = cudf.Series([-5, 7, 3, 4], index=list("abcd"))
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "dtype", ["datetime64[ns]", "timedelta64[ns]", "object", "str"]
)
def test_series_mixed_dtype_error(dtype):
    ps = pd.concat([pd.Series([1, 2, 3], dtype=dtype), pd.Series([10, 11])])
    with pytest.raises(TypeError):
        cudf.Series(ps)
    with pytest.raises(TypeError):
        cudf.Series(ps.array)


@pytest.mark.parametrize("data", [[True, False, None], [10, 200, 300]])
@pytest.mark.parametrize("index", [None, [10, 20, 30]])
def test_series_contains(data, index):
    ps = pd.Series(data, index=index)
    gs = cudf.Series(data, index=index)

    assert_eq(1 in ps, 1 in gs)
    assert_eq(10 in ps, 10 in gs)
    assert_eq(True in ps, True in gs)
    assert_eq(False in ps, False in gs)


def test_series_from_pandas_sparse():
    pser = pd.Series(range(2), dtype=pd.SparseDtype(np.int64, 0))
    with pytest.raises(NotImplementedError):
        cudf.Series(pser)


def test_series_constructor_unbounded_sequence():
    class A:
        def __getitem__(self, key):
            return 1

    with pytest.raises(TypeError):
        cudf.Series(A())


def test_series_constructor_error_mixed_type():
    with pytest.raises(MixedTypeError):
        cudf.Series(["abc", np.nan, "123"], nan_as_null=False)


def test_series_typecast_to_object_error():
    actual = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(ValueError):
            actual.astype(object)
        with pytest.raises(ValueError):
            actual.astype(np.dtype("object"))
        new_series = actual.astype("str")
        assert new_series[0] == "1970-01-01 00:00:00.000000001"


def test_series_typecast_to_object():
    actual = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with cudf.option_context("mode.pandas_compatible", False):
        new_series = actual.astype(object)
        assert new_series[0] == "1970-01-01 00:00:00.000000001"
        new_series = actual.astype(np.dtype("object"))
        assert new_series[0] == "1970-01-01 00:00:00.000000001"


@pytest.mark.parametrize("attr", ["nlargest", "nsmallest"])
def test_series_nlargest_nsmallest_str_error(attr):
    gs = cudf.Series(["a", "b", "c", "d", "e"])
    ps = gs.to_pandas()

    assert_exceptions_equal(
        getattr(gs, attr), getattr(ps, attr), ([], {"n": 1}), ([], {"n": 1})
    )


def test_series_unique_pandas_compatibility():
    gs = cudf.Series([10, 11, 12, 11, 10])
    ps = gs.to_pandas()
    with cudf.option_context("mode.pandas_compatible", True):
        actual = gs.unique()
    expected = ps.unique()
    assert_eq(actual, expected)


@pytest.mark.parametrize("initial_name", SERIES_OR_INDEX_NAMES)
@pytest.mark.parametrize("name", SERIES_OR_INDEX_NAMES)
def test_series_rename(initial_name, name):
    gsr = cudf.Series([1, 2, 3], name=initial_name)
    psr = pd.Series([1, 2, 3], name=initial_name)

    assert_eq(gsr, psr)

    actual = gsr.rename(name)
    expected = psr.rename(name)

    assert_eq(actual, expected)


@pytest.mark.parametrize("index", [lambda x: x * 2, {1: 2}])
def test_rename_index_not_supported(index):
    ser = cudf.Series(range(2))
    with pytest.raises(NotImplementedError):
        ser.rename(index=index)


@pytest.mark.parametrize(
    "data",
    [
        [1.2234242333234, 323432.3243423, np.nan],
        pd.Series([34224, 324324, 324342], dtype="datetime64[ns]"),
        pd.Series([224.242, None, 2424.234324], dtype="category"),
        [
            decimal.Decimal("342.3243234234242"),
            decimal.Decimal("89.32432497687622"),
            None,
        ],
    ],
)
@pytest.mark.parametrize("digits", [0, 1, 3, 4, 10])
def test_series_round_builtin(data, digits):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps, nan_as_null=False)

    # TODO: Remove `to_frame` workaround
    # after following issue is fixed:
    # https://github.com/pandas-dev/pandas/issues/55114
    expected = round(ps.to_frame(), digits)[0]
    expected.name = None
    actual = round(gs, digits)

    assert_eq(expected, actual)


def test_series_empty_dtype():
    expected = pd.Series([])
    actual = cudf.Series([])
    assert_eq(expected, actual, check_dtype=True)


@pytest.mark.parametrize("data", [None, {}, []])
def test_series_empty_index_rangeindex(data):
    expected = cudf.RangeIndex(0)
    result = cudf.Series(data).index
    assert_eq(result, expected)


def test_series_count_invalid_param():
    s = cudf.Series([], dtype="float64")
    with pytest.raises(TypeError):
        s.count(skipna=True)


@pytest.mark.parametrize(
    "data", [[0, 1, 2], ["a", "b", "c"], [0.324, 32.32, 3243.23]]
)
def test_series_setitem_nat_with_non_datetimes(data):
    s = cudf.Series(data)
    with pytest.raises(TypeError):
        s[0] = cudf.NaT


def test_series_string_setitem():
    gs = cudf.Series(["abc", "def", "ghi", "xyz", "pqr"])
    ps = gs.to_pandas()

    gs[0] = "NaT"
    gs[1] = "NA"
    gs[2] = "<NA>"
    gs[3] = "NaN"

    ps[0] = "NaT"
    ps[1] = "NA"
    ps[2] = "<NA>"
    ps[3] = "NaN"

    assert_eq(gs, ps)


def test_multi_dim_series_error():
    arr = cp.array([(1, 2), (3, 4)])
    with pytest.raises(ValueError):
        cudf.Series(arr)


def test_bool_series_mixed_dtype_error():
    ps = pd.Series([True, False, None])
    all_bool_ps = pd.Series([True, False, True], dtype="object")
    # ps now has `object` dtype, which
    # isn't supported by `cudf`.
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(TypeError):
            cudf.Series(ps)
        with pytest.raises(TypeError):
            cudf.from_pandas(ps)
        with pytest.raises(TypeError):
            cudf.Series(ps, dtype=bool)
        expected = cudf.Series(all_bool_ps, dtype=bool)
        assert_eq(expected, all_bool_ps.astype(bool))
    nan_bools_mix = pd.Series([True, False, True, np.nan], dtype="object")
    gs = cudf.Series(nan_bools_mix, nan_as_null=True)
    assert_eq(gs.to_pandas(nullable=True), nan_bools_mix.astype("boolean"))
    with pytest.raises(TypeError):
        cudf.Series(nan_bools_mix, nan_as_null=False)


@pytest.mark.parametrize(
    "pandas_type",
    [
        pd.ArrowDtype(pa.int8()),
        pd.ArrowDtype(pa.int16()),
        pd.ArrowDtype(pa.int32()),
        pd.ArrowDtype(pa.int64()),
        pd.ArrowDtype(pa.uint8()),
        pd.ArrowDtype(pa.uint16()),
        pd.ArrowDtype(pa.uint32()),
        pd.ArrowDtype(pa.uint64()),
        pd.ArrowDtype(pa.float32()),
        pd.ArrowDtype(pa.float64()),
        pd.Int8Dtype(),
        pd.Int16Dtype(),
        pd.Int32Dtype(),
        pd.Int64Dtype(),
        pd.UInt8Dtype(),
        pd.UInt16Dtype(),
        pd.UInt32Dtype(),
        pd.UInt64Dtype(),
        pd.Float32Dtype(),
        pd.Float64Dtype(),
    ],
)
def test_series_arrow_numeric_types_roundtrip(pandas_type):
    ps = pd.Series([1, 2, 3], dtype=pandas_type)
    pi = pd.Index(ps)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(ps)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pi)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pdf)


@pytest.mark.parametrize(
    "pandas_type", [pd.ArrowDtype(pa.bool_()), pd.BooleanDtype()]
)
def test_series_arrow_bool_types_roundtrip(pandas_type):
    ps = pd.Series([True, False, None], dtype=pandas_type)
    pi = pd.Index(ps)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(ps)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pi)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pdf)


@pytest.mark.parametrize(
    "pandas_type", [pd.ArrowDtype(pa.string()), pd.StringDtype()]
)
def test_series_arrow_string_types_roundtrip(pandas_type):
    ps = pd.Series(["abc", None, "xyz"], dtype=pandas_type)
    pi = pd.Index(ps)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(ps)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pi)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pdf)


def test_series_arrow_category_types_roundtrip():
    pa_array = pa.array(pd.Series([1, 2, 3], dtype="category"))
    ps = pd.Series([1, 2, 3], dtype=pd.ArrowDtype(pa_array.type))
    pi = pd.Index(ps)
    pdf = pi.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(ps)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pi)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pdf)


@pytest.mark.parametrize(
    "pa_type",
    [pa.decimal128(10, 2), pa.decimal128(5, 2), pa.decimal128(20, 2)],
)
def test_series_arrow_decimal_types_roundtrip(pa_type):
    ps = pd.Series(
        [
            decimal.Decimal("1.2"),
            decimal.Decimal("20.56"),
            decimal.Decimal("3"),
        ],
        dtype=pd.ArrowDtype(pa_type),
    )
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(ps)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pdf)


def test_series_arrow_struct_types_roundtrip():
    ps = pd.Series(
        [{"a": 1}, {"b": "abc"}],
        dtype=pd.ArrowDtype(pa.struct({"a": pa.int64(), "b": pa.string()})),
    )
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(ps)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pdf)


def test_series_arrow_list_types_roundtrip():
    ps = pd.Series([[1], [2], [4]], dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(ps)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(ps)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pdf)


@pytest.mark.parametrize("base_name", [None, "a"])
def test_series_to_frame_none_name(base_name):
    result = cudf.Series(range(1), name=base_name).to_frame(name=None)
    expected = pd.Series(range(1), name=base_name).to_frame(name=None)
    assert_eq(result, expected)


@pytest.mark.parametrize("klass", [cudf.Index, cudf.Series])
@pytest.mark.parametrize(
    "data", [pa.array([float("nan")]), pa.chunked_array([[float("nan")]])]
)
def test_nan_as_null_from_arrow_objects(klass, data):
    result = klass(data, nan_as_null=True)
    expected = klass(pa.array([None], type=pa.float64()))
    assert_eq(result, expected)


@pytest.mark.parametrize("reso", ["M", "ps"])
@pytest.mark.parametrize("typ", ["M", "m"])
def test_series_invalid_reso_dtype(reso, typ):
    with pytest.raises(TypeError):
        cudf.Series([], dtype=f"{typ}8[{reso}]")


def test_series_categorical_missing_value_count():
    ps = pd.Series(pd.Categorical(list("abcccb"), categories=list("cabd")))
    gs = cudf.from_pandas(ps)

    expected = ps.value_counts()
    actual = gs.value_counts()

    assert_eq(expected, actual, check_dtype=False)


def test_series_error_nan_mixed_types():
    ps = pd.Series([np.nan, "ab", "cd"])
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(MixedTypeError):
            cudf.from_pandas(ps)


def test_series_error_nan_non_float_dtypes():
    s = cudf.Series(["a", "b", "c"])
    with pytest.raises(TypeError):
        s[0] = np.nan

    s = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with pytest.raises(TypeError):
        s[0] = np.nan


@pytest.mark.parametrize(
    "dtype",
    [
        pd.ArrowDtype(pa.int8()),
        pd.ArrowDtype(pa.int16()),
        pd.ArrowDtype(pa.int32()),
        pd.ArrowDtype(pa.int64()),
        pd.ArrowDtype(pa.uint8()),
        pd.ArrowDtype(pa.uint16()),
        pd.ArrowDtype(pa.uint32()),
        pd.ArrowDtype(pa.uint64()),
        pd.ArrowDtype(pa.float32()),
        pd.ArrowDtype(pa.float64()),
        pd.Int8Dtype(),
        pd.Int16Dtype(),
        pd.Int32Dtype(),
        pd.Int64Dtype(),
        pd.UInt8Dtype(),
        pd.UInt16Dtype(),
        pd.UInt32Dtype(),
        pd.UInt64Dtype(),
        pd.Float32Dtype(),
        pd.Float64Dtype(),
    ],
)
@pytest.mark.parametrize("klass", [cudf.Series, cudf.DataFrame, cudf.Index])
@pytest.mark.parametrize("kind", [lambda x: x, str], ids=["obj", "string"])
def test_astype_pandas_nullable_pandas_compat(dtype, klass, kind):
    ser = klass([1, 2, 3])
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            ser.astype(kind(dtype))


@pytest.mark.parametrize("klass", [cudf.Series, cudf.Index])
@pytest.mark.parametrize(
    "data",
    [
        pa.array([1, None], type=pa.int64()),
        pa.chunked_array([[1, None]], type=pa.int64()),
    ],
)
def test_from_arrow_array_dtype(klass, data):
    obj = klass(data, dtype="int8")
    assert obj.dtype == np.dtype("int8")


@pytest.mark.parametrize("klass", [cudf.Series, cudf.Index])
def test_from_pandas_object_dtype_passed_dtype(klass):
    result = klass(pd.Series([True, False], dtype=object), dtype="int8")
    expected = klass(pa.array([1, 0], type=pa.int8()))
    assert_eq(result, expected)


def test_series_where_mixed_bool_dtype():
    s = cudf.Series([True, False, True])
    with pytest.raises(TypeError):
        s.where(~s, 10)


def test_series_setitem_mixed_bool_dtype():
    s = cudf.Series([True, False, True])
    with pytest.raises(TypeError):
        s[0] = 10


@pytest.mark.parametrize(
    "nat, value",
    [
        [np.datetime64("nat", "ns"), np.datetime64("2020-01-01", "ns")],
        [np.timedelta64("nat", "ns"), np.timedelta64(1, "ns")],
    ],
)
@pytest.mark.parametrize("nan_as_null", [True, False])
def test_series_np_array_nat_nan_as_nulls(nat, value, nan_as_null):
    expected = np.array([nat, value])
    ser = cudf.Series(expected, nan_as_null=nan_as_null)
    assert ser[0] is pd.NaT
    assert ser[1] == value


def test_series_unitness_np_datetimelike_units():
    data = np.array([np.timedelta64(1)])
    with pytest.raises(TypeError):
        cudf.Series(data)
    with pytest.raises(TypeError):
        pd.Series(data)


def test_series_duplicate_index_reindex():
    gs = cudf.Series([0, 1, 2, 3], index=[0, 0, 1, 1])
    ps = gs.to_pandas()

    assert_exceptions_equal(
        gs.reindex,
        ps.reindex,
        lfunc_args_and_kwargs=([10, 11, 12, 13], {}),
        rfunc_args_and_kwargs=([10, 11, 12, 13], {}),
    )


def test_list_category_like_maintains_dtype():
    dtype = cudf.CategoricalDtype(categories=[1, 2, 3, 4], ordered=True)
    data = [1, 2, 3]
    result = cudf.Series._from_column(
        cudf.core.column.as_column(data, dtype=dtype)
    )
    expected = pd.Series(data, dtype=dtype.to_pandas())
    assert_eq(result, expected)


def test_list_interval_like_maintains_dtype():
    dtype = cudf.IntervalDtype(subtype=np.int8)
    data = [pd.Interval(1, 2)]
    result = cudf.Series._from_column(
        cudf.core.column.as_column(data, dtype=dtype)
    )
    expected = pd.Series(data, dtype=dtype.to_pandas())
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "klass", [cudf.Series, cudf.Index, pd.Series, pd.Index]
)
def test_series_from_named_object_name_priority(klass):
    result = cudf.Series(klass([1], name="a"), name="b")
    assert result.name == "b"


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 3},
        cudf.Series([1, 2, 3], index=list("abc")),
        pd.Series([1, 2, 3], index=list("abc")),
    ],
)
def test_series_from_object_with_index_index_arg_reindex(data):
    result = cudf.Series(data, index=list("bca"))
    expected = cudf.Series([2, 3, 1], index=list("bca"))
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        {0: 1, 1: 2, 2: 3},
        cudf.Series([1, 2, 3]),
        cudf.Index([1, 2, 3]),
        pd.Series([1, 2, 3]),
        pd.Index([1, 2, 3]),
        [1, 2, 3],
    ],
)
def test_series_dtype_astypes(data):
    result = cudf.Series(data, dtype="float64")
    expected = cudf.Series([1.0, 2.0, 3.0])
    assert_eq(result, expected)


@pytest.mark.parametrize("pa_type", [pa.string, pa.large_string])
def test_series_from_large_string(pa_type):
    pa_string_array = pa.array(["a", "b", "c"]).cast(pa_type())
    got = cudf.Series(pa_string_array)
    expected = pd.Series(pa_string_array)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
        [1],
        decimal.Decimal("1.0"),
    ],
)
def test_series_to_pandas_arrow_type_nullable_raises(scalar):
    pa_array = pa.array([scalar, None])
    ser = cudf.Series(pa_array)
    with pytest.raises(ValueError, match=".* cannot both be set"):
        ser.to_pandas(nullable=True, arrow_type=True)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
        [1],
        decimal.Decimal("1.0"),
    ],
)
def test_series_to_pandas_arrow_type(scalar):
    pa_array = pa.array([scalar, None])
    ser = cudf.Series(pa_array)
    result = ser.to_pandas(arrow_type=True)
    expected = pd.Series(pd.arrays.ArrowExtensionArray(pa_array))
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, "index"])
@pytest.mark.parametrize("data", [[1, 2], [1]])
def test_squeeze(axis, data):
    ser = cudf.Series(data)
    result = ser.squeeze(axis=axis)
    expected = ser.to_pandas().squeeze(axis=axis)
    assert_eq(result, expected)


@pytest.mark.parametrize("axis", [1, "columns"])
def test_squeeze_invalid_axis(axis):
    with pytest.raises(ValueError):
        cudf.Series([1]).squeeze(axis=axis)


def test_series_init_with_nans():
    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.Series([1, 2, 3, np.nan])
    assert gs.dtype == np.dtype("float64")
    ps = pd.Series([1, 2, 3, np.nan])
    assert_eq(ps, gs)


@pytest.mark.parametrize("data", [None, 123, 33243243232423, 0])
def test_timestamp_series_init(data):
    scalar = pd.Timestamp(data)
    expected = pd.Series([scalar])
    actual = cudf.Series([scalar])

    assert_eq(expected, actual)

    expected = pd.Series(scalar)
    actual = cudf.Series(scalar)

    assert_eq(expected, actual)


@pytest.mark.parametrize("data", [None, 123, 33243243232423, 0])
def test_timedelta_series_init(data):
    scalar = pd.Timedelta(data)
    expected = pd.Series([scalar])
    actual = cudf.Series([scalar])

    assert_eq(expected, actual)

    expected = pd.Series(scalar)
    actual = cudf.Series(scalar)

    assert_eq(expected, actual)


def test_series_from_series_index_no_shallow_copy():
    ser1 = cudf.Series(range(3), index=list("abc"))
    ser2 = cudf.Series(ser1)
    assert ser1.index is ser2.index


@pytest.mark.parametrize("value", [1, 1.1])
def test_nans_to_nulls_noop_copies_column(value):
    ser1 = cudf.Series([value])
    ser2 = ser1.nans_to_nulls()
    assert ser1._column is not ser2._column


@pytest.mark.parametrize("dropna", [False, True])
def test_nunique_all_null(dropna):
    data = [None, None]
    pd_ser = pd.Series(data)
    cudf_ser = cudf.Series(data)
    result = pd_ser.nunique(dropna=dropna)
    expected = cudf_ser.nunique(dropna=dropna)
    assert result == expected


@pytest.mark.parametrize(
    "type1",
    [
        "category",
        "interval[int64, right]",
        "int64",
        "float64",
        "str",
        "datetime64[ns]",
        "timedelta64[ns]",
    ],
)
@pytest.mark.parametrize(
    "type2",
    [
        "category",
        "interval[int64, right]",
        "int64",
        "float64",
        "str",
        "datetime64[ns]",
        "timedelta64[ns]",
    ],
)
@pytest.mark.parametrize(
    "as_dtype", [lambda x: x, cudf.dtype], ids=["string", "object"]
)
@pytest.mark.parametrize("copy", [True, False])
def test_empty_astype_always_castable(type1, type2, as_dtype, copy):
    ser = cudf.Series([], dtype=as_dtype(type1))
    result = ser.astype(as_dtype(type2), copy=copy)
    expected = cudf.Series([], dtype=as_dtype(type2))
    assert_eq(result, expected)
    if not copy and cudf.dtype(type1) == cudf.dtype(type2):
        assert ser._column is result._column
    else:
        assert ser._column is not result._column


def test_dtype_dtypes_equal():
    ser = cudf.Series([0])
    assert ser.dtype is ser.dtypes
    assert ser.dtypes is ser.to_pandas().dtypes
