# Copyright (c) 2020-2025, NVIDIA CORPORATION.
import decimal
import operator
from collections import OrderedDict, defaultdict

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.core.column.column import as_column
from cudf.errors import MixedTypeError
from cudf.testing import assert_eq
from cudf.testing._utils import (
    NUMERIC_TYPES,
    SERIES_OR_INDEX_NAMES,
    assert_exceptions_equal,
    expect_warning_if,
    gen_rand,
)


@pytest.fixture(
    params=[
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
)
def ps(request):
    return request.param


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
    "data",
    [
        [0.25, 0.5, 0.2, -0.05],
        [0, 1, 2, np.nan, 4, cudf.NA, 6],
    ],
)
@pytest.mark.parametrize("lag", [1, 2, 3, 4])
def test_autocorr(data, lag):
    cudf_series = cudf.Series(data)
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
        [1, 2],
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        np.random.default_rng(seed=0).integers(-100, 100, 10),
        [],
        [np.nan, None, -1, 2, 3],
        [1.0, 12.0, None, None, 120],
        [0.1, 12.1, 14.1],
        [0, 14, 12, 12, 3, 10, 12, 14, None],
        [None, None, None],
        ["0", "12", "14"],
        ["0", "12", "14", "a"],
        [1.0, 2.5],
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


def test_fill_new_category():
    gs = cudf.Series(pd.Categorical(["a", "b", "c"]))
    with pytest.raises(TypeError):
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
        ["this", "is", None, "a", "test"],
        ["test", "this", "test", "is", None, "test", "a", "test"],
        ["0", "12", "14"],
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


def test_diff_unsupported_dtypes():
    gs = cudf.Series(["a", "b", "c", "d", "e"])
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
    assert s.dtype == np.dtype(f"i{default_integer_bitwidth // 8}")


@pytest.mark.parametrize("data", [[1.5, 2.5, 4.5], [1000, 2000, 4000, 3.14]])
def test_default_float_bitwidth_construction(default_float_bitwidth, data):
    s = cudf.Series(data)
    assert s.dtype == np.dtype(f"f{default_float_bitwidth // 8}")


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
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)

    with cudf.option_context("mode.pandas_compatible", True):
        gi = cudf.from_pandas(pi)
        assert_eq(pi, gi)

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "pandas_type", [pd.ArrowDtype(pa.bool_()), pd.BooleanDtype()]
)
def test_series_arrow_bool_types_roundtrip(pandas_type):
    ps = pd.Series([True, False, None], dtype=pandas_type)
    pi = pd.Index(ps)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)

    with cudf.option_context("mode.pandas_compatible", True):
        gi = cudf.from_pandas(pi)
        assert_eq(pi, gi)

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "pandas_type", [pd.ArrowDtype(pa.string()), pd.StringDtype()]
)
def test_series_arrow_string_types_roundtrip(pandas_type):
    ps = pd.Series(["abc", None, "xyz"], dtype=pandas_type)
    pi = pd.Index(ps)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)

    with cudf.option_context("mode.pandas_compatible", True):
        gi = cudf.from_pandas(pi)
        assert_eq(pi, gi)

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


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
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


def test_series_arrow_struct_types_roundtrip():
    ps = pd.Series(
        [{"a": 1}, {"b": "abc"}],
        dtype=pd.ArrowDtype(pa.struct({"a": pa.int64(), "b": pa.string()})),
    )
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


def test_series_arrow_list_types_roundtrip():
    ps = pd.Series([[1], [2], [4]], dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


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
        actual = ser.astype(kind(dtype))
        expected = klass([1, 2, 3], dtype=kind(dtype))
        assert_eq(actual, expected)


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


def test_series_duplicate_index_reindex():
    gs = cudf.Series([0, 1, 2, 3], index=[0, 0, 1, 1])
    ps = gs.to_pandas()

    assert_exceptions_equal(
        gs.reindex,
        ps.reindex,
        lfunc_args_and_kwargs=([10, 11, 12, 13], {}),
        rfunc_args_and_kwargs=([10, 11, 12, 13], {}),
    )


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


def test_null_like_to_nan_pandas_compat():
    with cudf.option_context("mode.pandas_compatible", True):
        ser = cudf.Series([1, 2, np.nan, 10, None])
        pser = pd.Series([1, 2, np.nan, 10, None])

        assert pser.dtype == ser.dtype
        assert_eq(ser, pser)


def test_roundtrip_series_plc_column(ps):
    expect = cudf.Series(ps)
    actual = cudf.Series.from_pylibcudf(*expect.to_pylibcudf())
    assert_eq(expect, actual)


def test_non_strings_dtype_object_pandas_compat_raises():
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(ValueError):
            cudf.Series([1], dtype=object)


def test_series_dataframe_count_float():
    gs = cudf.Series([1, 2, 3, None, np.nan, 10], nan_as_null=False)
    ps = cudf.Series([1, 2, 3, None, np.nan, 10])

    with cudf.option_context("mode.pandas_compatible", True):
        assert_eq(ps.count(), gs.count())
        assert_eq(ps.to_frame().count(), gs.to_frame().count())
    with cudf.option_context("mode.pandas_compatible", False):
        assert_eq(gs.count(), gs.to_pandas(nullable=True).count())
        assert_eq(
            gs.to_frame().count(),
            gs.to_frame().to_pandas(nullable=True).count(),
        )


@pytest.mark.parametrize("arr", [np.array, cp.array, pd.Series])
def test_construct_nonnative_array(arr):
    data = [1, 2, 3.5, 4]
    dtype = np.dtype("f4")
    native = arr(data, dtype=dtype)
    nonnative = arr(data, dtype=dtype.newbyteorder())
    result = cudf.Series(nonnative)
    expected = cudf.Series(native)
    assert_eq(result, expected)


@pytest.mark.parametrize("nan_as_null", [True, False])
def test_construct_all_pd_NA_with_dtype(nan_as_null):
    result = cudf.Series(
        [pd.NA, pd.NA], dtype=np.dtype(np.float64), nan_as_null=nan_as_null
    )
    expected = cudf.Series(pa.array([None, None], type=pa.float64()))
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "bool",
    ],
)
@pytest.mark.parametrize("has_nulls", [False, True])
@pytest.mark.parametrize("use_na_value", [False, True])
def test_series_to_cupy(dtype, has_nulls, use_na_value):
    size = 10
    if dtype == "bool":
        np_data = np.array([True, False] * (size // 2), dtype=bool)
    else:
        np_data = np.arange(size, dtype=dtype)

    if has_nulls:
        np_data = np_data.astype("object")
        np_data[::2] = None

    sr = cudf.Series(np_data, dtype=dtype)

    if not has_nulls:
        assert_eq(sr.values, cp.asarray(sr))
        return

    if has_nulls and not use_na_value:
        with pytest.raises(ValueError, match="Column must have no nulls"):
            sr.to_cupy()
        return

    na_value = {
        "bool": False,
        "float32": 0.0,
        "float64": 0.0,
    }.get(dtype, 0)
    expected = cp.asarray(sr.fillna(na_value)) if has_nulls else cp.asarray(sr)
    assert_eq(sr.to_cupy(na_value=na_value), expected)


def test_to_dense_array():
    rng = np.random.default_rng(seed=0)
    data = rng.random(8)
    mask = np.asarray([0b11010110]).astype(np.byte)
    sr = cudf.Series._from_column(
        as_column(data, dtype=np.float64).set_mask(mask)
    )
    assert sr.has_nulls
    assert sr.null_count != len(sr)
    filled = sr.to_numpy(na_value=np.nan)
    dense = sr.dropna().to_numpy()
    assert dense.size < filled.size
    assert filled.size == len(sr)
