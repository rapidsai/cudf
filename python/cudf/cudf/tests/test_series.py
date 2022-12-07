# Copyright (c) 2020-2022, NVIDIA CORPORATION.

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
from cudf.core._compat import PANDAS_GE_120, PANDAS_LT_140
from cudf.testing._utils import (
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
    _create_pandas_series,
    assert_eq,
    assert_exceptions_equal,
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
def test_series_append_basic(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = psr.append(other_ps, ignore_index=ignore_index)
    actual = gsr.append(other_gs, ignore_index=ignore_index)
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
def test_series_append_basic_str(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = psr.append(other_ps, ignore_index=ignore_index)
    actual = gsr.append(other_gs, ignore_index=ignore_index)
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
def test_series_append_series_with_index(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = cudf.from_pandas(others)

    expected = psr.append(other_ps, ignore_index=ignore_index)
    actual = gsr.append(other_gs, ignore_index=ignore_index)
    assert_eq(expected, actual)


def test_series_append_error_mixed_types():
    gsr = cudf.Series([1, 2, 3, 4])
    other = cudf.Series(["a", "b", "c", "d"])

    with pytest.raises(
        TypeError,
        match="cudf does not support mixed types, please type-cast "
        "both series to same dtypes.",
    ):
        gsr.append(other)

    with pytest.raises(
        TypeError,
        match="cudf does not support mixed types, please type-cast "
        "both series to same dtypes.",
    ):
        gsr.append([gsr, other, gsr, other])


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
def test_series_append_list_series_with_index(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = [cudf.from_pandas(obj) for obj in others]

    expected = psr.append(other_ps, ignore_index=ignore_index)
    actual = gsr.append(other_gs, ignore_index=ignore_index)
    assert_eq(expected, actual)


def test_series_append_existing_buffers():
    a1 = np.arange(10, dtype=np.float64)
    gs = cudf.Series(a1)

    # Add new buffer
    a2 = cudf.Series(np.arange(5))
    gs = gs.append(a2)
    assert len(gs) == 15
    np.testing.assert_equal(gs.to_numpy(), np.hstack([a1, a2.to_numpy()]))

    # Ensure appending to previous buffer
    a3 = cudf.Series(np.arange(3))
    gs = gs.append(a3)
    assert len(gs) == 18
    a4 = np.hstack([a1, a2.to_numpy(), a3.to_numpy()])
    np.testing.assert_equal(gs.to_numpy(), a4)

    # Appending different dtype
    a5 = cudf.Series(np.array([1, 2, 3], dtype=np.int32))
    a6 = cudf.Series(np.array([4.5, 5.5, 6.5], dtype=np.float64))
    gs = a5.append(a6)
    np.testing.assert_equal(
        gs.to_numpy(), np.hstack([a5.to_numpy(), a6.to_numpy()])
    )
    gs = cudf.Series(a6).append(a5)
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
    psr = _create_pandas_series(data)
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
    expected = ps.describe(datetime_is_numeric=True)
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
@pytest.mark.parametrize("na_sentinel", [99999, 11, -1, 0])
def test_series_factorize(data, na_sentinel):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas()

    expected_labels, expected_cats = psr.factorize(na_sentinel=na_sentinel)
    actual_labels, actual_cats = gsr.factorize(na_sentinel=na_sentinel)

    assert_eq(expected_labels, actual_labels.get())
    assert_eq(expected_cats.values, actual_cats.to_pandas().values)


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

    if len(data) > 0:
        if nulls == "one":
            p = np.random.randint(0, len(data))
            psr[p] = None
        elif nulls == "some":
            p = np.random.randint(0, len(data), 2)
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
    np.random.seed(12)
    pd_cat = pd.Categorical(
        pd.Series(
            np.random.choice(list(ascii_letters + digits), num_elements),
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
    for size in [10**x for x in range(5)]:
        arr = np.random.randint(low=-1, high=10, size=size)
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

        assert_eq(expect, got, check_dtype=False, check_index_type=False)


@pytest.mark.parametrize("bins", [1, 2, 3])
def test_series_value_counts_bins(bins):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(bins=bins)
    got = gsr.value_counts(bins=bins)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=False)


@pytest.mark.parametrize("bins", [1, 2, 3])
@pytest.mark.parametrize("dropna", [True, False])
def test_series_value_counts_bins_dropna(bins, dropna):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, np.nan])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(bins=bins, dropna=dropna)
    got = gsr.value_counts(bins=bins, dropna=dropna)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=False)


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

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=False)
    assert_eq(
        expected.reset_index(drop=True),
        got.reset_index(drop=True),
        check_dtype=False,
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
        cudf.Series(name="empty series"),
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
        np.random.normal(-100, 100, 1000),
        np.random.randint(-50, 50, 1000),
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

    # with nulls, maintaining existing null mask
    arr = arr.astype("float64")  # for pandas nulls
    arr.ravel()[
        np.random.choice(arr.shape[0], arr.shape[0] // 2, replace=False)
    ] = np.nan

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

    gs = cudf.Series.from_pandas(ps, nan_as_null=nan_as_null)

    assert_eq(ps.isnull(), gs.isnull())
    assert_eq(ps.isna(), gs.isna())


@pytest.mark.parametrize("ps", _series_na_data())
@pytest.mark.parametrize("nan_as_null", [True, False, None])
def test_series_notnull_notna(ps, nan_as_null):

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
    ],
)
@pytest.mark.parametrize(
    "labels",
    [[1], [0], 1, 5, [5, 9], pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
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
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=([], {}),
        rfunc_args_and_kwargs=([], {}),
        expected_error_message="Need to specify at least one",
    )

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=(["b"], {"axis": 1}),
        rfunc_args_and_kwargs=(["b"], {"axis": 1}),
        expected_error_message="No axis named 1",
    )


def test_series_drop_raises():
    gs = cudf.Series([10, 20, 30], index=["x", "y", "z"], name="c")
    ps = gs.to_pandas()

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=(["p"],),
        rfunc_args_and_kwargs=(["p"],),
        expected_error_message="One or more values not found in axis",
    )

    # dtype specified mismatch
    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=([3],),
        rfunc_args_and_kwargs=([3],),
        expected_error_message="One or more values not found in axis",
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
def test_series_upcast_float16(data):
    actual_series = cudf.Series(data)
    expected_series = cudf.Series(data, dtype="float32")
    assert_eq(actual_series, expected_series)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(0, 3, 1),
        [3.0, 1.0, np.nan],
        ["a", "z", None],
        pytest.param(
            pd.RangeIndex(4, -1, -2),
            marks=[
                pytest.mark.xfail(
                    condition=PANDAS_LT_140,
                    reason="https://github.com/pandas-dev/pandas/issues/43591",
                )
            ],
        ),
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


@pytest.mark.parametrize("method", ["md5"])
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
    s = cudf.Series()
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
@pytest.mark.parametrize("name", [None, "ser"])
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
        expected_error_message="occurs multiple times, use a level number",
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
        np.random.randint(-100, 100, 200),
        pd.Series([0.0, 1.0, None, 10.0]),
        [None, None, None, None],
        [np.nan, None, -1, 2, 3],
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        np.random.randint(-100, 100, 10),
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
    index = np.random.randint(0, 100, len(data))
    psr = _create_pandas_series(data, index=index)
    gsr = cudf.Series.from_pandas(psr, nan_as_null=False)

    expected = psr.isin(values)
    got = gsr.isin(values)

    assert_eq(got, expected)


@pytest.mark.xfail(raises=TypeError)
def test_fill_new_category():
    gs = cudf.Series(pd.Categorical(["a", "b", "c"]))
    gs[0:1] = "d"


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
    psr = _create_pandas_series(data)
    gsr = cudf.Series.from_pandas(psr)

    got = gsr.isin(values)
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
        pytest.param(
            [12, 14, 19],
            marks=pytest.mark.xfail(
                not PANDAS_GE_120,
                reason="pandas's failure here seems like a bug(in < 1.2) "
                "given the reverse succeeds",
            ),
        ),
        ["is", "this", "is", "this", "is"],
    ],
)
def test_isin_string(data, values):
    psr = _create_pandas_series(data)
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
    psr = _create_pandas_series(data)
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
@pytest.mark.parametrize("dtype", NUMERIC_TYPES + ["bool"])
@pytest.mark.parametrize("series_bins", [True, False])
def test_series_digitize(num_rows, num_bins, right, dtype, series_bins):
    data = np.random.randint(0, 100, num_rows).astype(dtype)
    bins = np.unique(np.sort(np.random.randint(2, 95, num_bins).astype(dtype)))
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
    s = cudf.Series(np.random.randint(0, 30, 80), dtype="int32")
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
    sr = cudf.Series(np.random.randint(0, 100, 1000))
    # pandas unique() preserves order
    expect = pd.Series(sr.to_pandas().unique())
    got = cudf.Series(sr._column.unique(preserve_order=True))
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

    assert_eq(gs.hasnans, ps.hasnans)


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
def test_series_duplicated(data, index, keep):
    gs = cudf.Series(data, index=index)
    ps = gs.to_pandas()

    assert_eq(gs.duplicated(keep=keep), ps.duplicated(keep=keep))
