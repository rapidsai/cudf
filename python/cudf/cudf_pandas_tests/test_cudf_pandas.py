# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import copy
import datetime
import operator
import pathlib
import pickle
import tempfile
import types
from io import BytesIO, StringIO

import numpy as np
import pyarrow as pa
import pytest
from numba import NumbaDeprecationWarning
from pytz import utc

from cudf.pandas import LOADED, Profiler
from cudf.pandas.fast_slow_proxy import _Unusable

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

import pandas as xpd
import pandas._testing as tm
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    EasterMonday,
    GoodFriday,
    Holiday,
    USColumbusDay,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    get_calendar,
)

# Accelerated pandas has the real pandas module as an attribute
pd = xpd._fsproxy_slow


@pytest.fixture
def dataframe():
    pdf = pd.DataFrame({"a": [1, 1, 1, 2, 3], "b": [1, 2, 3, 4, 5]})
    df = xpd.DataFrame(pdf)
    return (pdf, df)


@pytest.fixture
def series(dataframe):
    pdf, df = dataframe
    return (pdf["a"], df["a"])


@pytest.fixture
def index():
    return (
        pd.Index(["a", "b", "c", "d", "e"]),
        xpd.Index(["a", "b", "c", "d", "e"]),
    )


@pytest.fixture
def multiindex(dataframe):
    pdf, df = dataframe
    pmi = pd.MultiIndex.from_frame(pdf)
    mi = xpd.MultiIndex.from_frame(df)
    return (pmi, mi)


@pytest.fixture
def array(series):
    arr, xarr = series
    return (arr.values, xarr.values)


@pytest.fixture(
    params=[
        lambda group: group["a"].sum(),
        lambda group: group.sum().apply(lambda val: [val]),
    ]
)
def groupby_udf(request):
    return request.param


def test_assert_equal():
    tm.assert_frame_equal(
        pd.DataFrame({"a": [1, 2, 3]}), xpd.DataFrame({"a": [1, 2, 3]})
    )

    with pytest.raises(AssertionError):
        tm.assert_frame_equal(
            pd.DataFrame({"a": [1, 2, 3]}), xpd.DataFrame({"a": [1, 2, 4]})
        )


def test_construction():
    # test that constructing a DataFrame returns an DataFrame
    data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
    pdf = pd.DataFrame(data)
    df = xpd.DataFrame(data)
    tm.assert_frame_equal(pdf, df)


def test_construction_object():
    # test that we can construct a Series with `object` dtype
    psr = pd.Series([1, "a", [1, 2, 3]])
    sr = xpd.Series([1, "a", [1, 2, 3]])
    tm.assert_series_equal(psr, sr)


def test_construction_from_frame(dataframe):
    pdf, _ = dataframe
    df = xpd.DataFrame(pdf)
    tm.assert_frame_equal(pdf, df)


def test_groupby(dataframe):
    pdf, df = dataframe
    expected = pdf.groupby("a", sort=True).max()
    gb = df.groupby("a", sort=True)
    got = gb.max()
    tm.assert_frame_equal(expected, got)


def test_repr(dataframe):
    pdf, df = dataframe
    assert df.__repr__() == pdf.__repr__()


def test_binops_series(series):
    psr, sr = series
    expected = psr + psr
    got = sr + sr
    tm.assert_series_equal(expected, got)


def test_binops_df(dataframe):
    pdf, df = dataframe
    expected = pdf + pdf
    got = df + df
    tm.assert_frame_equal(expected, got)


def test_attribute(dataframe):
    pdf, df = dataframe
    assert pdf.shape == df.shape


def test_tz_localize():
    psr = pd.Series(["2001-01-01", "2002-02-02"], dtype="datetime64[ms]")
    sr = xpd.Series(psr)
    tm.assert_series_equal(
        psr.dt.tz_localize("America/New_York"),
        sr.dt.tz_localize("America/New_York"),
        check_dtype=False,
    )


def test_index_tz_localize():
    pti = pd.Index(pd.date_range("2020-01-01", periods=3, freq="D"))
    xti = xpd.Index(xpd.date_range("2020-01-01", periods=3, freq="D"))
    pti = pti.tz_localize("UTC")
    xti = xti.tz_localize("UTC")
    tm.assert_equal(pti, xti)


def test_index_generator():
    pi = pd.Index(iter(range(10)))
    xi = xpd.Index(iter(range(10)))
    tm.assert_equal(pi, xi)


def test_groupby_apply_fallback(dataframe, groupby_udf):
    pdf, df = dataframe
    tm.assert_equal(
        pdf.groupby("a", sort=True, group_keys=True).apply(groupby_udf),
        df.groupby("a", sort=True, group_keys=True).apply(groupby_udf),
    )


def test_groupby_external_series_apply_fallback(dataframe, groupby_udf):
    pdf, df = dataframe
    tm.assert_equal(
        pdf.groupby(
            pd.Series([1, 2, 1, 2, 1]), sort=True, group_keys=True
        ).apply(groupby_udf),
        df.groupby(
            xpd.Series([1, 2, 1, 2, 1]), sort=True, group_keys=True
        ).apply(groupby_udf),
    )


def test_read_csv():
    data = "1,2,3\n4,5,6"
    expected = pd.read_csv(StringIO(data))
    got = xpd.read_csv(StringIO(data))
    tm.assert_frame_equal(expected, got)


def test_iloc(dataframe):
    pdf, df = dataframe
    tm.assert_frame_equal(pdf.iloc[:, :], df.iloc[:, :])


def test_neg(dataframe):
    pdf, df = dataframe
    tm.assert_frame_equal(-pdf, -df)


def test_groupby_filter(dataframe):
    pdf, df = dataframe
    expected = pdf.groupby("a").filter(lambda df: len(df) > 2)
    got = df.groupby("a").filter(lambda df: len(df) > 2)
    tm.assert_frame_equal(expected, got)


def test_groupby_rolling(dataframe):
    pdf, df = dataframe
    expected = pdf.groupby("a").rolling(2).sum()
    got = df.groupby("a").rolling(2).sum()
    tm.assert_frame_equal(expected, got)


def test_groupby_rolling_window(dataframe):
    pdf, df = dataframe
    expected = pdf.groupby("a").rolling(2, win_type="triang").mean()
    got = df.groupby("a").rolling(2, win_type="triang").mean()
    tm.assert_frame_equal(expected, got)


def test_ewm():
    pdf = pd.DataFrame(range(5))
    df = xpd.DataFrame(range(5))
    result = df.ewm(0.5).mean()
    expected = pdf.ewm(0.5).mean()
    tm.assert_equal(result, expected)


def test_setitem_frame(dataframe):
    pdf, df = dataframe
    pdf[pdf > 1] = -pdf
    df[df > 1] = -df
    tm.assert_frame_equal(pdf, df)


def test_concat(dataframe):
    pdf, df = dataframe
    expected = pd.concat([pdf, pdf])
    got = xpd.concat([df, df])
    tm.assert_frame_equal(expected, got)


def test_attribute_error():
    df = xpd.DataFrame()
    with pytest.raises(AttributeError):
        df.blah


def test_df_from_series(series):
    psr, sr = series
    tm.assert_frame_equal(pd.DataFrame(psr), xpd.DataFrame(sr))


def test_iloc_change_type(series):
    psr, sr = series
    psr.iloc[0] = "a"
    sr.iloc[0] = "a"
    tm.assert_series_equal(psr, sr)


def test_rename_categories():
    psr = pd.Series([1, 2, 3], dtype="category")
    sr = xpd.Series([1, 2, 3], dtype="category")
    psr = psr.cat.rename_categories({1: 5})
    sr = sr.cat.rename_categories({1: 5})
    tm.assert_series_equal(psr, sr)


def test_column_rename(dataframe):
    pdf, df = dataframe
    pdf.columns = ["x", "y"]
    df.columns = ["x", "y"]
    tm.assert_frame_equal(pdf, df)


def test_shape(dataframe):
    pdf, df = dataframe
    assert pdf.shape == df.shape
    pdf["c"] = range(5)
    df["c"] = range(5)
    assert pdf.shape == df.shape


def test_isnull():
    psr = pd.Series([1, 2, 3])
    sr = xpd.Series(psr)
    # Test that invoking `Pandas` functions works.
    tm.assert_series_equal(pd.isnull(psr), xpd.isnull(sr))


def test_copy_deepcopy_recursion(dataframe):
    # test that we don't recurse when calling the copy/deepcopy
    # methods, which can happen due to
    # https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
    import copy

    pdf, df = dataframe
    copy.copy(df)
    copy.deepcopy(df)


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_copy_deepcopy(copier):
    s = xpd.Series([1, 2, 3])
    s2 = copier(s)
    assert isinstance(s2, s.__class__)
    tm.assert_equal(s, s2)

    df = xpd.DataFrame({"a": [1, 2, 3]})
    df2 = copier(df)
    assert isinstance(df2, df.__class__)
    tm.assert_equal(df, df2)

    idx = xpd.Index([1, 2, 3])
    idx2 = copier(idx)
    assert isinstance(idx2, idx.__class__)
    tm.assert_equal(idx, idx2)


def test_classmethod():
    pdf = pd.DataFrame.from_dict({"a": [1, 2, 3]})
    df = xpd.DataFrame.from_dict({"a": [1, 2, 3]})
    tm.assert_frame_equal(pdf, df)


def test_rolling(dataframe):
    pdf, df = dataframe
    tm.assert_frame_equal(pdf.rolling(2).agg("sum"), df.rolling(2).agg("sum"))


def test_array_function_series(series):
    psr, sr = series
    np.testing.assert_allclose(np.average(psr), np.average(sr))


def test_array_function_ndarray(array):
    arr, xarr = array
    np.isclose(np.average(arr), np.average(xarr))


def test_histogram_ndarray(array):
    arr, xarr = array
    expected_hist, expected_edges = np.histogram(arr, bins="auto")
    got_hist, got_edges = np.histogram(xarr, bins="auto")
    tm.assert_almost_equal(expected_hist, got_hist)
    tm.assert_almost_equal(expected_edges, got_edges)


def test_pickle_round_trip(dataframe):
    pdf, df = dataframe
    pickled_pdf = BytesIO()
    pickled_cudf_pandas = BytesIO()
    pdf.to_pickle(pickled_pdf)
    df.to_pickle(pickled_cudf_pandas)

    pickled_pdf.seek(0)
    pickled_cudf_pandas.seek(0)

    tm.assert_frame_equal(
        pd.read_pickle(pickled_pdf), xpd.read_pickle(pickled_cudf_pandas)
    )


def test_excel_round_trip(dataframe):
    pdf, df = dataframe
    excel_pdf = BytesIO()
    excel_cudf_pandas = BytesIO()
    pdf.to_excel(excel_pdf)
    df.to_excel(excel_cudf_pandas)

    excel_pdf.seek(0)
    excel_cudf_pandas.seek(0)

    tm.assert_frame_equal(
        pd.read_excel(excel_pdf), xpd.read_excel(excel_cudf_pandas)
    )


def test_hash_array(series):
    ps, xs = series
    expected = pd.util.hash_array(ps.values)
    actual = xpd.util.hash_array(xs.values)

    tm.assert_almost_equal(expected, actual)


def test_is_sparse():
    psa = pd.arrays.SparseArray([0, 0, 1, 0])
    xsa = xpd.arrays.SparseArray([0, 0, 1, 0])

    assert pd.api.types.is_sparse(psa) == xpd.api.types.is_sparse(xsa)


def test_is_file_like():
    assert pd.api.types.is_file_like("a") == xpd.api.types.is_file_like("a")
    assert pd.api.types.is_file_like(BytesIO()) == xpd.api.types.is_file_like(
        BytesIO()
    )
    assert pd.api.types.is_file_like(
        StringIO("abc")
    ) == xpd.api.types.is_file_like(StringIO("abc"))


def test_is_re_compilable():
    assert pd.api.types.is_re_compilable(
        ".^"
    ) == xpd.api.types.is_re_compilable(".^")
    assert pd.api.types.is_re_compilable(
        ".*"
    ) == xpd.api.types.is_re_compilable(".*")


def test_module_attribute_types():
    assert isinstance(xpd.read_csv, types.FunctionType)
    assert isinstance(xpd.tseries.frequencies.Day, type)
    assert isinstance(xpd.api, types.ModuleType)


def test_infer_freq():
    expected = pd.infer_freq(
        pd.date_range(start="2020/12/01", end="2020/12/30", periods=30)
    )
    got = xpd.infer_freq(
        xpd.date_range(start="2020/12/01", end="2020/12/30", periods=30)
    )
    assert expected == got


def test_groupby_grouper_fallback(dataframe, groupby_udf):
    pdf, df = dataframe
    tm.assert_equal(
        pdf.groupby(pd.Grouper("a"), sort=True, group_keys=True).apply(
            groupby_udf
        ),
        df.groupby(xpd.Grouper("a"), sort=True, group_keys=True).apply(
            groupby_udf
        ),
    )


def test_options_mode():
    assert xpd.options.mode.copy_on_write == pd.options.mode.copy_on_write


def test_profiler():
    pytest.importorskip("cudf")

    # test that the profiler correctly reports
    # when we use the GPU v/s CPU
    with Profiler() as p:
        df = xpd.DataFrame({"a": [1, 2, 3], "b": "b"})
        df.groupby("a").max()

    assert len(p.per_line_stats) == 2
    for line_no, line, gpu_time, cpu_time in p.per_line_stats:
        assert gpu_time
        assert not cpu_time

    with Profiler() as p:
        s = xpd.Series([1, "a"])
        s = s + s

    assert len(p.per_line_stats) == 2
    for line_no, line, gpu_time, cpu_time in p.per_line_stats:
        assert cpu_time


def test_column_access_as_attribute():
    pdf = pd.DataFrame({"fast": [1, 2, 3], "slow": [2, 3, 4]})
    df = xpd.DataFrame({"fast": [1, 2, 3], "slow": [2, 3, 4]})

    tm.assert_series_equal(pdf.fast, df.fast)
    tm.assert_series_equal(pdf.slow, df.slow)


def test_binop_dataframe_list(dataframe):
    pdf, df = dataframe
    expect = pdf[["a"]] == [[1, 2, 3, 4, 5]]
    got = df[["a"]] == [[1, 2, 3, 4, 5]]
    tm.assert_frame_equal(expect, got)


def test_binop_array_series(series):
    psr, sr = series
    arr = psr.array
    expect = arr + psr
    got = arr + sr
    tm.assert_series_equal(expect, got)


def test_array_ufunc_reduction(series):
    psr, sr = series
    expect = np.ufunc.reduce(np.subtract, psr)
    got = np.ufunc.reduce(np.subtract, sr)
    tm.assert_equal(expect, got)


def test_array_ufunc(series):
    psr, sr = series
    expect = np.subtract(psr, psr)
    got = np.subtract(sr, sr)
    assert isinstance(got, sr.__class__)
    tm.assert_equal(expect, got)


@pytest.mark.xfail(strict=False, reason="Fails in CI, passes locally.")
def test_groupby_apply_func_returns_series(dataframe):
    pdf, df = dataframe
    expect = pdf.groupby("a").apply(
        lambda group: pd.Series({"x": 1}), include_groups=False
    )
    got = df.groupby("a").apply(
        lambda group: xpd.Series({"x": 1}), include_groups=False
    )
    tm.assert_equal(expect, got)


@pytest.mark.parametrize("data", [[1, 2, 3], ["a", None, "b"]])
def test_pyarrow_array_construction(data):
    cudf_pandas_series = xpd.Series(data)
    actual_pa_array = pa.array(cudf_pandas_series)
    expected_pa_array = pa.array(data)
    assert actual_pa_array.equals(expected_pa_array)


@pytest.mark.parametrize(
    "op", [">", "<", "==", "<=", ">=", "+", "%", "-", "*", "/"]
)
def test_cudf_pandas_eval_series(op):
    lhs = xpd.Series([10, 11, 12])  # noqa: F841
    rhs = xpd.Series([100, 1, 12])  # noqa: F841

    actual = xpd.eval(f"lhs {op} rhs")

    pd_lhs = pd.Series([10, 11, 12])  # noqa: F841
    pd_rhs = pd.Series([100, 1, 12])  # noqa: F841

    expected = pd.eval(f"pd_lhs {op} pd_rhs")

    tm.assert_series_equal(expected, actual)


@pytest.mark.parametrize(
    "op", [">", "<", "==", "<=", ">=", "+", "%", "-", "*", "/"]
)
def test_cudf_pandas_eval_dataframe(op):
    lhs = xpd.DataFrame({"a": [10, 11, 12], "b": [1, 2, 3]})  # noqa: F841
    rhs = xpd.DataFrame({"a": [100, 1, 12], "b": [15, -10, 3]})  # noqa: F841

    actual = xpd.eval(f"lhs {op} rhs")

    pd_lhs = pd.DataFrame({"a": [10, 11, 12], "b": [1, 2, 3]})  # noqa: F841
    pd_rhs = pd.DataFrame({"a": [100, 1, 12], "b": [15, -10, 3]})  # noqa: F841

    expected = pd.eval(f"pd_lhs {op} pd_rhs")

    tm.assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "expr", ["((a + b) * c % d) > e", "((a + b) * c % d)"]
)
def test_cudf_pandas_eval_complex(expr):
    data = {
        "a": [10, 11, 12],
        "b": [1, 2, 3],
        "c": [100, 1, 12],
        "d": [15, -10, 3],
        "e": [100, 200, 300],
    }
    cudf_pandas_frame = xpd.DataFrame(data)
    pd_frame = pd.DataFrame(data)
    actual = cudf_pandas_frame.eval(expr)
    expected = pd_frame.eval(expr)
    tm.assert_series_equal(expected, actual)


def test_array_function_series_fallback(series):
    psr, sr = series
    expect = np.unique(psr, return_counts=True)
    got = np.unique(sr, return_counts=True)
    tm.assert_equal(expect, got)


def test_timedeltaproperties(series):
    psr, sr = series
    psr, sr = psr.astype("timedelta64[ns]"), sr.astype("timedelta64[ns]")
    tm.assert_equal(psr.dt.days, sr.dt.days)
    tm.assert_equal(psr.dt.components, sr.dt.components)
    tm.assert_equal(psr.dt.total_seconds(), sr.dt.total_seconds())


@pytest.mark.parametrize("scalar_type", [int, float, complex, bool])
@pytest.mark.parametrize("scalar", [1, 1.0, True, 0])
def test_coerce_zero_d_array_to_scalar(scalar_type, scalar):
    expected = scalar_type(pd.Series([scalar]).values[0])
    got = scalar_type(xpd.Series([scalar]).values[0])
    tm.assert_equal(expected, got)


def test_cupy_asarray_zero_copy():
    cp = pytest.importorskip("cupy")

    sr = xpd.Series([1, 2, 3])
    cpary = cp.asarray(sr.values)

    assert (
        sr.__cuda_array_interface__["data"][0]
        == cpary.__cuda_array_interface__["data"][0]
    )


def test_pipe(dataframe):
    pdf, df = dataframe

    def func(df, x):
        return df + x

    expect = pdf.pipe(func, 1)
    got = df.pipe(func, 1)
    tm.assert_frame_equal(expect, got)


def test_pipe_tuple(dataframe):
    pdf, df = dataframe

    def func(x, df):
        return df + x

    expect = pdf.pipe((func, "df"), 1)
    got = df.pipe((func, "df"), 1)
    tm.assert_frame_equal(expect, got)


def test_maintain_container_subclasses(multiindex):
    # pandas Frozenlist is a list subclass
    pmi, mi = multiindex
    got = mi.names.difference(["b"])
    expect = pmi.names.difference(["b"])
    assert got == expect
    assert isinstance(got, xpd.core.indexes.frozen.FrozenList)


def test_rolling_win_type():
    pdf = pd.DataFrame(range(5))
    df = xpd.DataFrame(range(5))
    result = df.rolling(2, win_type="boxcar").mean()
    expected = pdf.rolling(2, win_type="boxcar").mean()
    tm.assert_equal(result, expected)


@pytest.mark.skip(
    reason="Requires Numba 0.59 to fix segfaults on ARM. See https://github.com/numba/llvmlite/pull/1009"
)
def test_rolling_apply_numba_engine():
    def weighted_mean(x):
        arr = np.ones((1, x.shape[1]))
        arr[:, :2] = (x[:, :2] * x[:, 2]).sum(axis=0) / x[:, 2].sum()
        return arr

    pdf = pd.DataFrame([[1, 2, 0.6], [2, 3, 0.4], [3, 4, 0.2], [4, 5, 0.7]])
    df = xpd.DataFrame([[1, 2, 0.6], [2, 3, 0.4], [3, 4, 0.2], [4, 5, 0.7]])

    with pytest.warns(NumbaDeprecationWarning):
        expect = pdf.rolling(2, method="table", min_periods=0).apply(
            weighted_mean, raw=True, engine="numba"
        )
    got = df.rolling(2, method="table", min_periods=0).apply(
        weighted_mean, raw=True, engine="numba"
    )
    tm.assert_equal(expect, got)


def test_expanding():
    pdf = pd.DataFrame(range(5))
    df = xpd.DataFrame(range(5))
    result = df.expanding().mean()
    expected = pdf.expanding().mean()
    tm.assert_equal(result, expected)


def test_pipe_with_data_creating_func():
    def pandas_func(df):
        df2 = pd.DataFrame({"b": np.arange(len(df))})
        return df.join(df2)

    def cudf_pandas_func(df):
        df2 = xpd.DataFrame({"b": np.arange(len(df))})
        return df.join(df2)

    pdf = pd.DataFrame({"a": [1, 2, 3]})
    df = xpd.DataFrame({"a": [1, 2, 3]})

    tm.assert_frame_equal(pdf.pipe(pandas_func), df.pipe(cudf_pandas_func))


@pytest.mark.parametrize(
    "data",
    [
        '{"a": 1, "b": 2, "c": 3}',
        '{"a": 1, "b": 2, "c": 3}\n{"a": 4, "b": 5, "c": 6}',
    ],
)
def test_chunked_json_reader(tmpdir, data):
    file_path = tmpdir / "test.json"
    with open(file_path, "w") as f:
        f.write(data)

    with (
        pd.read_json(file_path, lines=True, chunksize=1) as pd_reader,
        xpd.read_json(file_path, lines=True, chunksize=1) as xpd_reader,
    ):
        for pd_chunk, xpd_chunk in zip(pd_reader, xpd_reader):
            tm.assert_equal(pd_chunk, xpd_chunk)

    with (
        pd.read_json(StringIO(data), lines=True, chunksize=1) as pd_reader,
        xpd.read_json(StringIO(data), lines=True, chunksize=1) as xpd_reader,
    ):
        for pd_chunk, xpd_chunk in zip(pd_reader, xpd_reader):
            tm.assert_equal(pd_chunk, xpd_chunk)


@pytest.mark.parametrize(
    "data",
    [
        "1,2,3",
        "1,2,3\n4,5,6",
    ],
)
def test_chunked_csv_reader(tmpdir, data):
    file_path = tmpdir / "test.json"
    with open(file_path, "w") as f:
        f.write(data)

    with (
        pd.read_csv(file_path, chunksize=1) as pd_reader,
        xpd.read_csv(file_path, chunksize=1) as xpd_reader,
    ):
        for pd_chunk, xpd_chunk in zip(pd_reader, xpd_reader):
            tm.assert_equal(pd_chunk, xpd_chunk, check_index_type=False)

    with (
        pd.read_json(StringIO(data), lines=True, chunksize=1) as pd_reader,
        xpd.read_json(StringIO(data), lines=True, chunksize=1) as xpd_reader,
    ):
        for pd_chunk, xpd_chunk in zip(pd_reader, xpd_reader):
            tm.assert_equal(pd_chunk, xpd_chunk, check_index_type=False)


@pytest.mark.parametrize(
    "data", [(), (1,), (1, 2, 3), ("a", "b", "c"), (1, 2, "test")]
)
def test_construct_from_generator(data):
    expect = pd.Series((x for x in data))
    got = xpd.Series((x for x in data))
    tm.assert_series_equal(expect, got)


def test_read_csv_stringio_usecols():
    data = "col1,col2,col3\na,b,1\na,b,2\nc,d,3"
    expect = pd.read_csv(StringIO(data), usecols=lambda x: x.upper() != "COL3")
    got = xpd.read_csv(StringIO(data), usecols=lambda x: x.upper() != "COL3")
    tm.assert_frame_equal(expect, got)


def test_construct_datetime_index():
    expect = pd.DatetimeIndex([10, 20, 30], dtype="datetime64[ns]")
    got = xpd.DatetimeIndex([10, 20, 30], dtype="datetime64[ns]")
    tm.assert_index_equal(expect, got)


def test_construct_timedelta_index():
    expect = pd.TimedeltaIndex([10, 20, 30], dtype="timedelta64[ns]")
    got = xpd.TimedeltaIndex([10, 20, 30], dtype="timedelta64[ns]")
    tm.assert_index_equal(expect, got)


@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.sub,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ],
)
def test_datetime_ops(op):
    pd_dt_idx1 = pd.DatetimeIndex([10, 20, 30], dtype="datetime64[ns]")
    cudf_pandas_dt_idx = xpd.DatetimeIndex(
        [10, 20, 30], dtype="datetime64[ns]"
    )

    tm.assert_equal(
        op(pd_dt_idx1, pd_dt_idx1), op(cudf_pandas_dt_idx, cudf_pandas_dt_idx)
    )


@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.add,
        operator.sub,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ],
)
def test_timedelta_ops(op):
    pd_td_idx1 = pd.TimedeltaIndex([10, 20, 30], dtype="timedelta64[ns]")
    cudf_pandas_td_idx = xpd.TimedeltaIndex(
        [10, 20, 30], dtype="timedelta64[ns]"
    )

    tm.assert_equal(
        op(pd_td_idx1, pd_td_idx1), op(cudf_pandas_td_idx, cudf_pandas_td_idx)
    )


@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_datetime_timedelta_ops(op):
    pd_dt_idx1 = pd.DatetimeIndex([10, 20, 30], dtype="datetime64[ns]")
    cudf_pandas_dt_idx = xpd.DatetimeIndex(
        [10, 20, 30], dtype="datetime64[ns]"
    )

    pd_td_idx1 = pd.TimedeltaIndex([10, 20, 30], dtype="timedelta64[ns]")
    cudf_pandas_td_idx = xpd.TimedeltaIndex(
        [10, 20, 30], dtype="timedelta64[ns]"
    )

    tm.assert_equal(
        op(pd_dt_idx1, pd_td_idx1), op(cudf_pandas_dt_idx, cudf_pandas_td_idx)
    )


def test_itertuples():
    df = xpd.DataFrame(range(1))
    result = next(iter(df.itertuples()))
    tup = collections.namedtuple("Pandas", ["Index", "1"], rename=True)
    expected = tup(0, 0)
    assert result == expected
    assert result._fields == expected._fields


def test_namedagg_namedtuple():
    df = xpd.DataFrame(
        {
            "kind": ["cat", "dog", "cat", "dog"],
            "height": [9.1, 6.0, 9.5, 34.0],
            "weight": [7.9, 7.5, 9.9, 198.0],
        }
    )
    result = df.groupby("kind").agg(
        min_height=pd.NamedAgg(column="height", aggfunc="min"),
        max_height=pd.NamedAgg(column="height", aggfunc="max"),
        average_weight=pd.NamedAgg(column="weight", aggfunc=np.mean),
    )
    expected = xpd.DataFrame(
        {
            "min_height": [9.1, 6.0],
            "max_height": [9.5, 34.0],
            "average_weight": [8.90, 102.75],
        },
        index=xpd.Index(["cat", "dog"], name="kind"),
    )
    tm.assert_frame_equal(result, expected)


def test_dataframe_dir(dataframe):
    """Test that column names are present in the dataframe dir

    We do not test direct dir equality because pandas does some runtime
    modifications of dir that we cannot replicate without forcing D2H
    conversions (e.g. modifying what elements are visible based on the contents
    of a DataFrame instance).
    """
    _, df = dataframe
    assert "a" in dir(df)
    assert "b" in dir(df)

    # Only string types are added to dir
    df[1] = [1] * len(df)
    assert 1 not in dir(df)


def test_array_copy(array):
    arr, xarr = array
    tm.assert_equal(copy.copy(arr), copy.copy(xarr))


def test_datetime_values_dtype_roundtrip():
    s = pd.Series([1, 2, 3], dtype="datetime64[ns]")
    xs = xpd.Series([1, 2, 3], dtype="datetime64[ns]")
    expected = np.asarray(s.values)
    actual = np.asarray(xs.values)
    assert expected.dtype == actual.dtype
    tm.assert_equal(expected, actual)


def test_resample():
    ser = pd.Series(
        range(3), index=pd.date_range("2020-01-01", freq="D", periods=3)
    )
    xser = xpd.Series(
        range(3), index=xpd.date_range("2020-01-01", freq="D", periods=3)
    )
    expected = ser.resample("D").max()
    result = xser.resample("D").max()
    # TODO: See if as_unit can be avoided
    expected.index = expected.index.as_unit("s")
    result.index = result.index.as_unit("s")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("accessor", ["str", "dt", "cat"])
def test_accessor_types(accessor):
    assert isinstance(getattr(xpd.Series, accessor), type)


@pytest.mark.parametrize(
    "op",
    [operator.itemgetter(1), operator.methodcaller("sum")],
    ids=["getitem[1]", ".sum()"],
)
def test_values_zero_dim_result_is_scalar(op):
    s = pd.Series([1, 2, 3])
    x = xpd.Series([1, 2, 3])
    expect = op(s.values)
    got = op(x.values)
    assert expect == got
    assert type(expect) is type(got)


@pytest.mark.parametrize("box", ["DataFrame", "Series"])
def test_round_builtin(box):
    xobj = getattr(xpd, box)([1.23])
    pobj = getattr(pd, box)([1.23])
    result = round(xobj, 1)
    expected = round(pobj, 1)
    tm.assert_equal(result, expected)


def test_hash():
    xobj = xpd.Timedelta(1)
    pobj = pd.Timedelta(1)
    assert hash(xobj) == hash(pobj)


def test_non_hashable_object():
    with pytest.raises(TypeError):
        hash(pd.DataFrame(range(1)))

    with pytest.raises(TypeError):
        hash(xpd.DataFrame(range(1)))


@pytest.mark.parametrize("offset", ["DateOffset", "Day", "BDay"])
def test_timestamp_offset_binop(offset):
    ts = xpd.Timestamp(2020, 1, 1)
    result = ts + getattr(xpd.offsets, offset)()
    expected = pd.Timestamp(2020, 1, 2)
    tm.assert_equal(result, expected)


def test_string_dtype():
    xobj = xpd.StringDtype()
    pobj = pd.StringDtype()
    tm.assert_equal(xobj, pobj)


def test_string_array():
    data = np.array(["1"], dtype=object)
    xobj = xpd.arrays.StringArray(data)
    pobj = pd.arrays.StringArray(data)
    tm.assert_extension_array_equal(xobj, pobj)


def test_subclass_series():
    class foo(pd.Series):
        def __init__(self, myinput):
            super().__init__(myinput)

    s1 = pd.Series([1, 2, 3])
    s2 = foo(myinput=[1, 2, 3])

    tm.assert_equal(s1, s2, check_series_type=False)


@pytest.mark.parametrize(
    "index_type",
    [
        xpd.RangeIndex,
        xpd.CategoricalIndex,
        xpd.DatetimeIndex,
        xpd.TimedeltaIndex,
        xpd.PeriodIndex,
        xpd.MultiIndex,
        xpd.IntervalIndex,
    ],
)
def test_index_subclass(index_type):
    # test that proxy index types are derived
    # from Index
    assert issubclass(index_type, xpd.Index)
    assert not issubclass(xpd.Index, index_type)


def test_np_array_of_timestamps():
    expected = np.array([pd.Timestamp(1)]) + pd.tseries.offsets.MonthEnd()
    got = np.array([xpd.Timestamp(1)]) + xpd.tseries.offsets.MonthEnd()
    tm.assert_equal(expected, got)


@pytest.mark.parametrize(
    "obj",
    [
        # Basic types
        xpd.Series(dtype="float64"),
        xpd.Series([1, 2, 3]),
        xpd.DataFrame(dtype="float64"),
        xpd.DataFrame({"a": [1, 2, 3]}),
        xpd.Series([1, 2, 3]),
        # Index (doesn't support nullary construction)
        xpd.Index([1, 2, 3]),
        xpd.Index(["a", "b", "c"]),
        # Complex index
        xpd.to_datetime(
            [
                "1/1/2018",
                np.datetime64("2018-01-01"),
                datetime.datetime(2018, 1, 1),
            ]
        ),
        # Objects where the underlying store is the slow type.
        xpd.Series(["a", 2, 3]),
        xpd.Index(["a", 2, 3]),
        # Other types
        xpd.tseries.offsets.BDay(5),
        xpd.Timestamp("2001-01-01"),
        xpd.Timestamp("2001-01-01", tz="UTC"),
        xpd.Timedelta("1 days"),
        xpd.Timedelta(1, "D"),
    ],
)
def test_pickle(obj):
    with tempfile.TemporaryFile() as f:
        pickle.dump(obj, f)
        f.seek(0)
        copy = pickle.load(f)

    tm.assert_equal(obj, copy)


def test_dataframe_query():
    cudf_pandas_df = xpd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    pd_df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})

    actual = cudf_pandas_df.query("foo > 2")
    expected = pd_df.query("foo > 2")

    tm.assert_equal(actual, expected)

    bizz = 2  # noqa: F841
    actual = cudf_pandas_df.query("foo > @bizz")
    expected = pd_df.query("foo > @bizz")

    tm.assert_equal(actual, expected)


def test_private_method_result_wrapped():
    xoffset = xpd.offsets.Day()
    dt = datetime.datetime(2020, 1, 1)
    result = xoffset._apply(dt)
    assert isinstance(result, xpd.Timestamp)


def test_numpy_var():
    np.random.seed(42)
    data = np.random.rand(1000)
    psr = pd.Series(data)
    sr = xpd.Series(data)

    tm.assert_almost_equal(np.var(psr), np.var(sr))


def test_index_new():
    expected = pd.Index.__new__(pd.Index, [1, 2, 3])
    got = xpd.Index.__new__(xpd.Index, [1, 2, 3])
    tm.assert_equal(expected, got)

    expected = pd.Index.__new__(pd.Index, [1, 2, 3], dtype="int8")
    got = xpd.Index.__new__(xpd.Index, [1, 2, 3], dtype="int8")
    tm.assert_equal(expected, got)

    expected = pd.RangeIndex.__new__(pd.RangeIndex, 0, 10, 2)
    got = xpd.RangeIndex.__new__(xpd.RangeIndex, 0, 10, 2)
    tm.assert_equal(expected, got)


@pytest.mark.xfail(not LOADED, reason="Should not fail in accelerated mode")
def test_groupby_apply_callable_referencing_pandas(dataframe):
    pdf, df = dataframe

    class Callable1:
        def __call__(self, df):
            if not isinstance(df, pd.DataFrame):
                raise TypeError
            return 1

    class Callable2:
        def __call__(self, df):
            if not isinstance(df, xpd.DataFrame):
                raise TypeError
            return 1

    expect = pdf.groupby("a").apply(Callable1())
    got = df.groupby("a").apply(Callable2())

    tm.assert_equal(expect, got)


def test_constructor_properties(dataframe, series, index):
    _, df = dataframe
    _, sr = series
    _, idx = index

    assert df._constructor is xpd.DataFrame
    assert sr._constructor is xpd.Series
    assert idx._constructor is xpd.Index
    assert sr._constructor_expanddim is xpd.DataFrame
    assert df._constructor_sliced is xpd.Series


def test_pos():
    xser = +xpd.Series([-1])
    ser = +pd.Series([-1])
    tm.assert_equal(xser, ser)


def test_intermediates_are_proxied():
    df = xpd.DataFrame({"a": [1, 2, 3]})
    grouper = df.groupby("a")
    assert isinstance(grouper, xpd.core.groupby.generic.DataFrameGroupBy)


def test_from_dataframe():
    cudf = pytest.importorskip("cudf")
    from cudf.testing._utils import assert_eq

    data = {"foo": [1, 2, 3], "bar": [4, 5, 6]}

    cudf_pandas_df = xpd.DataFrame(data)
    cudf_df = cudf.DataFrame(data)

    # test construction of a cuDF DataFrame from an cudf_pandas DataFrame
    assert_eq(cudf_df, cudf.DataFrame.from_pandas(cudf_pandas_df))
    assert_eq(cudf_df, cudf.from_dataframe(cudf_pandas_df))

    # ideally the below would work as well, but currently segfaults

    # pd_df = pd.DataFrame(data)
    # assert_eq(pd_df, pd.api.interchange.from_dataframe(cudf_pandas_df))


def test_multiindex_values_returns_1d_tuples():
    mi = xpd.MultiIndex.from_tuples([(1, 2), (3, 4)])
    result = mi.values
    expected = np.empty(2, dtype=object)
    expected[...] = [(1, 2), (3, 4)]
    tm.assert_equal(result, expected)


def test_read_sas_context():
    cudf_path = pathlib.Path(__file__).parent.parent
    path = cudf_path / "cudf" / "tests" / "data" / "sas" / "cars.sas7bdat"
    with xpd.read_sas(path, format="sas7bdat", iterator=True) as reader:
        df = reader.read()
    assert isinstance(df, xpd.DataFrame)


def test_concat_fast():
    pytest.importorskip("cudf")

    assert type(xpd.concat._fsproxy_fast) is not _Unusable


def test_func_namespace():
    # note: this test is sensitive to Pandas' internal module layout
    assert xpd.concat is xpd.core.reshape.concat.concat


def test_pickle_groupby(dataframe):
    pdf, df = dataframe
    pgb = pdf.groupby("a")
    gb = df.groupby("a")
    gb = pickle.loads(pickle.dumps(gb))
    tm.assert_equal(pgb.sum(), gb.sum())


def test_numpy_extension_array():
    np_array = np.array([0, 1, 2, 3])
    xarray = xpd.arrays.NumpyExtensionArray(np_array)
    array = pd.arrays.NumpyExtensionArray(np_array)

    tm.assert_equal(xarray, array)


def test_isinstance_base_offset():
    offset = xpd.tseries.frequencies.to_offset("1s")
    assert isinstance(offset, xpd.tseries.offsets.BaseOffset)


def test_floordiv_array_vs_df():
    xarray = xpd.Series([1, 2, 3], dtype="datetime64[ns]").array
    parray = pd.Series([1, 2, 3], dtype="datetime64[ns]").array

    xdf = xpd.DataFrame(xarray)
    pdf = pd.DataFrame(parray)

    actual = xarray.__floordiv__(xdf)
    expected = parray.__floordiv__(pdf)

    tm.assert_equal(actual, expected)


def test_apply_slow_path_udf_references_global_module():
    def my_apply(df, unused):
        # `datetime` Raised `KeyError: __import__`
        datetime.datetime.strptime(df["Minute"], "%H:%M:%S")
        return pd.to_numeric(1)

    df = xpd.DataFrame({"Minute": ["09:00:00"]})
    result = df.apply(my_apply, axis=1, unused=True)
    expected = xpd.Series([1])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        "__iadd__",
        "__iand__",
        "__ifloordiv__",
        "__imod__",
        "__imul__",
        "__ior__",
        "__ipow__",
        "__isub__",
        "__itruediv__",
        "__ixor__",
    ],
)
def test_inplace_ops(op):
    xdf1 = xpd.DataFrame({"a": [10, 11, 12]})
    xdf2 = xpd.DataFrame({"a": [1, 2, 3]})

    df1 = pd.DataFrame({"a": [10, 11, 12]})
    df2 = pd.DataFrame({"a": [1, 2, 3]})

    actual = getattr(xdf1, op)(xdf2)
    expected = getattr(df1, op)(df2)

    tm.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "op",
    [
        "__iadd__",
        "__iand__",
        "__ifloordiv__",
        "__imod__",
        "__imul__",
        "__ior__",
        "__ipow__",
        "__isub__",
        "__itruediv__",
        "__ixor__",
    ],
)
def test_inplace_ops_series(op):
    xser1 = xpd.Series([10, 11, 12])
    xser2 = xpd.Series([1, 2, 3])

    ser1 = pd.Series([10, 11, 12])
    ser2 = pd.Series([1, 2, 3])

    actual = getattr(xser1, op)(xser2)
    expected = getattr(ser1, op)(ser2)

    tm.assert_equal(actual, expected)


@pytest.mark.parametrize("data", [pd.NaT, 1234, "nat"])
def test_timestamp(data):
    xtimestamp = xpd.Timestamp(data)
    timestamp = pd.Timestamp(data)
    tm.assert_equal(xtimestamp, timestamp)


@pytest.mark.parametrize("data", [pd.NaT, 1234, "nat"])
def test_timedelta(data):
    xtimedelta = xpd.Timedelta(data)
    timedelta = pd.Timedelta(data)
    tm.assert_equal(xtimedelta, timedelta)


def test_abstract_holiday_calendar():
    class TestCalendar(AbstractHolidayCalendar):
        def __init__(self, name=None, rules=None) -> None:
            super().__init__(name=name, rules=rules)

    jan1 = TestCalendar(rules=[Holiday("jan1", year=2015, month=1, day=1)])
    jan2 = TestCalendar(rules=[Holiday("jan2", year=2015, month=1, day=2)])

    # Getting holidays for Jan 1 should not alter results for Jan 2.
    expected = xpd.DatetimeIndex(["01-Jan-2015"]).as_unit("ns")
    tm.assert_index_equal(jan1.holidays(), expected)

    expected2 = xpd.DatetimeIndex(["02-Jan-2015"]).as_unit("ns")
    tm.assert_index_equal(jan2.holidays(), expected2)


@pytest.mark.parametrize(
    "holiday,start,expected",
    [
        (USMemorialDay, datetime.datetime(2015, 7, 1), []),
        (USLaborDay, "2015-09-07", [xpd.Timestamp("2015-09-07")]),
        (USColumbusDay, "2015-10-12", [xpd.Timestamp("2015-10-12")]),
        (USThanksgivingDay, "2015-11-26", [xpd.Timestamp("2015-11-26")]),
        (USMartinLutherKingJr, "2015-01-19", [xpd.Timestamp("2015-01-19")]),
        (USPresidentsDay, datetime.datetime(2015, 7, 1), []),
        (GoodFriday, datetime.datetime(2015, 7, 1), []),
        (EasterMonday, "2015-04-06", [xpd.Timestamp("2015-04-06")]),
        ("New Year's Day", "2010-12-31", [xpd.Timestamp("2010-12-31")]),
        ("Independence Day", "2015-07-03", [xpd.Timestamp("2015-07-03")]),
        ("Veterans Day", "2012-11-11", []),
        ("Christmas Day", "2011-12-26", [xpd.Timestamp("2011-12-26")]),
        (
            "Juneteenth National Independence Day",
            "2021-06-18",
            [xpd.Timestamp("2021-06-18")],
        ),
        ("Juneteenth National Independence Day", "2022-06-19", []),
        (
            "Juneteenth National Independence Day",
            "2022-06-20",
            [xpd.Timestamp("2022-06-20")],
        ),
    ],
)
def test_holidays_within_dates(holiday, start, expected):
    if isinstance(holiday, str):
        calendar = get_calendar("USFederalHolidayCalendar")
        holiday = calendar.rule_from_name(holiday)

    assert list(holiday.dates(start, start)) == expected

    # Verify that timezone info is preserved.
    assert list(
        holiday.dates(
            utc.localize(xpd.Timestamp(start)),
            utc.localize(xpd.Timestamp(start)),
        )
    ) == [utc.localize(dt) for dt in expected]
