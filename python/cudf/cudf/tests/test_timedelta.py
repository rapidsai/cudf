# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import datetime
import operator
import re

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_120
from cudf.testing import _utils as utils
from cudf.testing._utils import assert_eq, assert_exceptions_equal

_TIMEDELTA_DATA = [
    [1000000, 200000, 3000000],
    [1000000, 200000, None],
    [],
    [None],
    [None, None, None, None, None],
    [12, 12, 22, 343, 4353534, 435342],
    np.array([10, 20, 30, None, 100]),
    cp.asarray([10, 20, 30, 100]),
    [1000000, 200000, 3000000],
    [1000000, 200000, None],
    [1],
    [12, 11, 232, 223432411, 2343241, 234324, 23234],
    [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
    [
        136457654736252,
        134736784364431,
        245345345545332,
        223432411,
        2343241,
        3634548734,
        23234,
    ],
    [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
]

_TIMEDELTA_DATA_NON_OVERFLOW = [
    [1000000, 200000, 3000000],
    [1000000, 200000, None],
    [],
    [None],
    [None, None, None, None, None],
    [12, 12, 22, 343, 4353534, 435342],
    np.array([10, 20, 30, None, 100]),
    cp.asarray([10, 20, 30, 100]),
    [1000000, 200000, 3000000],
    [1000000, 200000, None],
    [1],
    [12, 11, 232, 223432411, 2343241, 234324, 23234],
    [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
    [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
]


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        [0.3534, 12, 22, 343, 43.53534, 4353.42],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_series_create(data, dtype):
    if dtype not in ("timedelta64[ns]"):
        pytest.skip(
            "Bug in pandas : https://github.com/pandas-dev/pandas/issues/35465"
        )
    psr = pd.Series(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data, dtype=dtype
    )
    gsr = cudf.Series(data, dtype=dtype)

    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [12, 12, 22, 343, 4353534, 435342],
        [0.3534, 12, 22, 343, 43.53534, 4353.42],
        cp.asarray([10, 20, 30, 100]),
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize("cast_dtype", ["int64", "category"])
def test_timedelta_from_typecast(data, dtype, cast_dtype):
    if dtype not in ("timedelta64[ns]"):
        pytest.skip(
            "Bug in pandas : https://github.com/pandas-dev/pandas/issues/35465"
        )
    psr = pd.Series(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data, dtype=dtype
    )
    gsr = cudf.Series(data, dtype=dtype)

    if cast_dtype == "int64":
        assert_eq(psr.values.view(cast_dtype), gsr.astype(cast_dtype).values)
    else:
        assert_eq(psr.astype(cast_dtype), gsr.astype(cast_dtype))


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [12, 12, 22, 343, 4353534, 435342],
        [0.3534, 12, 22, 343, 43.53534, 4353.42],
        cp.asarray([10, 20, 30, 100]),
    ],
)
@pytest.mark.parametrize("cast_dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_to_typecast(data, cast_dtype):
    psr = pd.Series(cp.asnumpy(data) if isinstance(data, cp.ndarray) else data)
    gsr = cudf.Series(data)

    assert_eq(psr.astype(cast_dtype), gsr.astype(cast_dtype))


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        [0.3534, 12, 22, 343, 43.53534, 4353.42],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_from_pandas(data, dtype):
    psr = pd.Series(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data, dtype=dtype
    )
    gsr = cudf.from_pandas(psr)

    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_series_to_numpy(data, dtype):
    gsr = cudf.Series(data, dtype=dtype)

    expected = np.array(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data, dtype=dtype
    )
    expected = expected[~np.isnan(expected)]

    actual = gsr.dropna().to_numpy()

    np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_series_to_pandas(data, dtype):
    gsr = cudf.Series(data, dtype=dtype)

    expected = np.array(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data, dtype=dtype
    )

    expected = pd.Series(expected)
    actual = gsr.to_pandas()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data,other",
    [
        ([1000000, 200000, 3000000], [1000000, 200000, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, None]),
        ([], []),
        ([None], [None]),
        ([None, None, None, None, None], [None, None, None, None, None]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [12, 12, 22, 343, 4353534, 435342],
        ),
        (np.array([10, 20, 30, None, 100]), np.array([10, 20, 30, None, 100])),
        (cp.asarray([10, 20, 30, 100]), cp.asarray([10, 20, 30, 100])),
        ([1000000, 200000, 3000000], [200000, 34543, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, 3000000]),
        ([None], [1]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [None, 1, 220, 3, 34, 4353423287],
        ),
        (np.array([10, 20, 30, None, 100]), np.array([10, 20, 30, None, 100])),
        (cp.asarray([10, 20, 30, 100]), cp.asarray([10, 20, 30, 100])),
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize(
    "ops",
    [
        "eq",
        "ne",
        "lt",
        "gt",
        "le",
        "ge",
        "add",
        "radd",
        "sub",
        "rsub",
        "floordiv",
        "truediv",
        "mod",
    ],
)
def test_timedelta_ops_misc_inputs(data, other, dtype, ops):
    gsr = cudf.Series(data, dtype=dtype)
    other_gsr = cudf.Series(other, dtype=dtype)

    psr = gsr.to_pandas()
    other_psr = other_gsr.to_pandas()

    expected = getattr(psr, ops)(other_psr)
    actual = getattr(gsr, ops)(other_gsr)
    if ops in ("eq", "lt", "gt", "le", "ge"):
        actual = actual.fillna(False)
    elif ops == "ne":
        actual = actual.fillna(True)

    if ops == "floordiv":
        expected[actual.isna().to_pandas()] = np.nan

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "datetime_data,timedelta_data",
    [
        ([1000000, 200000, 3000000], [1000000, 200000, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, None]),
        ([], []),
        ([None], [None]),
        ([None, None, None, None, None], [None, None, None, None, None]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [12, 12, 22, 343, 4353534, 435342],
        ),
        (np.array([10, 20, 30, None, 100]), np.array([10, 20, 30, None, 100])),
        (cp.asarray([10, 20, 30, 100]), cp.asarray([10, 20, 30, 100])),
        ([1000000, 200000, 3000000], [200000, 34543, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, 3000000]),
        ([None], [1]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [None, 1, 220, 3, 34, 4353423287],
        ),
        (np.array([10, 20, 30, None, 100]), np.array([10, 20, 30, None, 100])),
        (cp.asarray([10, 20, 30, 100]), cp.asarray([10, 20, 30, 100])),
        (
            [12, 11, 232, 223432411, 2343241, 234324, 23234],
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
        ),
        (
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
        ),
        (
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        ),
        (
            [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        ),
    ],
)
@pytest.mark.parametrize("datetime_dtype", utils.DATETIME_TYPES)
@pytest.mark.parametrize("timedelta_dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize(
    "ops",
    ["add", "sub"],
)
def test_timedelta_ops_datetime_inputs(
    datetime_data, timedelta_data, datetime_dtype, timedelta_dtype, ops
):
    gsr_datetime = cudf.Series(datetime_data, dtype=datetime_dtype)
    gsr_timedelta = cudf.Series(timedelta_data, dtype=timedelta_dtype)

    psr_datetime = gsr_datetime.to_pandas()
    psr_timedelta = gsr_timedelta.to_pandas()

    expected = getattr(psr_datetime, ops)(psr_timedelta)
    actual = getattr(gsr_datetime, ops)(gsr_timedelta)

    assert_eq(expected, actual)

    if ops == "add":
        expected = getattr(psr_timedelta, ops)(psr_datetime)
        actual = getattr(gsr_timedelta, ops)(gsr_datetime)

        assert_eq(expected, actual)
    elif ops == "sub":
        assert_exceptions_equal(
            lfunc=operator.sub,
            rfunc=operator.sub,
            lfunc_args_and_kwargs=([psr_timedelta, psr_datetime],),
            rfunc_args_and_kwargs=([gsr_timedelta, gsr_datetime],),
            compare_error_message=False,
        )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {
                "A": pd.Series(pd.date_range("2012-1-1", periods=3, freq="D")),
                "B": pd.Series([pd.Timedelta(days=i) for i in range(3)]),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series(
                    pd.date_range("1994-1-1", periods=50, freq="D")
                ),
                "B": pd.Series([pd.Timedelta(days=i) for i in range(50)]),
            }
        ),
    ],
)
@pytest.mark.parametrize("op", ["add", "sub"])
def test_timedelta_dataframe_ops(df, op):
    pdf = df
    gdf = cudf.from_pandas(pdf)

    if op == "add":
        pdf["C"] = pdf["A"] + pdf["B"]
        gdf["C"] = gdf["A"] + gdf["B"]
    elif op == "sub":
        pdf["C"] = pdf["A"] - pdf["B"]
        gdf["C"] = gdf["A"] - gdf["B"]

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        pytest.param(
            [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/5938"
            ),
        ),
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "other_scalars",
    [
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        datetime.timedelta(minutes=447),
        datetime.timedelta(hours=447),
        datetime.timedelta(weeks=734),
        np.timedelta64(4, "s"),
        np.timedelta64(456, "D"),
        np.timedelta64(46, "h"),
        pytest.param(
            np.timedelta64("nat"),
            marks=pytest.mark.xfail(
                condition=not PANDAS_GE_120,
                reason="https://github.com/pandas-dev/pandas/issues/35529",
            ),
        ),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize(
    "op",
    [
        "add",
        "sub",
        "truediv",
        "mod",
        pytest.param(
            "floordiv",
            marks=pytest.mark.xfail(
                condition=not PANDAS_GE_120,
                reason="https://github.com/pandas-dev/pandas/issues/35529",
            ),
        ),
    ],
)
def test_timedelta_series_ops_with_scalars(data, other_scalars, dtype, op):
    gsr = cudf.Series(data=data, dtype=dtype)
    psr = gsr.to_pandas()

    if op == "add":
        expected = psr + other_scalars
        actual = gsr + other_scalars
    elif op == "sub":
        expected = psr - other_scalars
        actual = gsr - other_scalars
    elif op == "truediv":
        expected = psr / other_scalars
        actual = gsr / other_scalars
    elif op == "floordiv":
        expected = psr // other_scalars
        actual = gsr // other_scalars
    elif op == "mod":
        expected = psr % other_scalars
        actual = gsr % other_scalars

    assert_eq(expected, actual)

    if op == "add":
        expected = other_scalars + psr
        actual = other_scalars + gsr
    elif op == "sub":
        expected = other_scalars - psr
        actual = other_scalars - gsr
    elif op == "truediv":
        expected = other_scalars / psr
        actual = other_scalars / gsr
    elif op == "floordiv":
        expected = other_scalars // psr
        actual = other_scalars // gsr
    elif op == "mod":
        expected = other_scalars % psr
        actual = other_scalars % gsr

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        pytest.param(
            [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/5938"
            ),
        ),
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "cpu_scalar",
    [
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        np.timedelta64(4, "s"),
        pytest.param(
            np.timedelta64("nat", "s"),
            marks=pytest.mark.xfail(
                condition=not PANDAS_GE_120,
                reason="https://github.com/pandas-dev/pandas/issues/35529",
            ),
        ),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64("nat", "ns"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize(
    "op",
    [
        "add",
        "sub",
        "truediv",
        "mod",
        pytest.param(
            "floordiv",
            marks=pytest.mark.xfail(
                condition=not PANDAS_GE_120,
                reason="https://github.com/pandas-dev/pandas/issues/35529",
            ),
        ),
    ],
)
def test_timedelta_series_ops_with_cudf_scalars(data, cpu_scalar, dtype, op):
    gpu_scalar = cudf.Scalar(cpu_scalar)

    gsr = cudf.Series(data=data, dtype=dtype)
    psr = gsr.to_pandas()

    if op == "add":
        expected = psr + cpu_scalar
        actual = gsr + gpu_scalar
    elif op == "sub":
        expected = psr - cpu_scalar
        actual = gsr - gpu_scalar
    elif op == "truediv":
        expected = psr / cpu_scalar
        actual = gsr / gpu_scalar
    elif op == "floordiv":
        expected = psr // cpu_scalar
        actual = gsr // gpu_scalar
    elif op == "mod":
        expected = psr % cpu_scalar
        actual = gsr % gpu_scalar

    assert_eq(expected, actual)

    if op == "add":
        expected = cpu_scalar + psr
        actual = gpu_scalar + gsr
    elif op == "sub":
        expected = cpu_scalar - psr
        actual = gpu_scalar - gsr
    elif op == "truediv":
        expected = cpu_scalar / psr
        actual = gpu_scalar / gsr
    elif op == "floordiv":
        expected = cpu_scalar // psr
        actual = gpu_scalar // gsr
    elif op == "mod":
        expected = cpu_scalar % psr
        actual = gpu_scalar % gsr

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize("reduction_op", ["sum", "mean", "median", "quantile"])
def test_timedelta_reduction_ops(data, dtype, reduction_op):
    gsr = cudf.Series(data, dtype=dtype)
    psr = gsr.to_pandas()

    if len(psr) > 0 and psr.isnull().all() and reduction_op == "median":
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            expected = getattr(psr, reduction_op)()
    else:
        expected = getattr(psr, reduction_op)()
    actual = getattr(gsr, reduction_op)()
    if pd.isna(expected) and pd.isna(actual):
        pass
    elif isinstance(expected, pd.Timedelta) and isinstance(
        actual, pd.Timedelta
    ):
        assert (
            expected.round(gsr._column.time_unit).value
            == actual.round(gsr._column.time_unit).value
        )
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    _TIMEDELTA_DATA,
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_dt_components(data, dtype):
    gsr = cudf.Series(data, dtype=dtype)
    psr = gsr.to_pandas()

    expected = psr.dt.components
    actual = gsr.dt.components

    if gsr.isnull().any():
        assert_eq(expected, actual.astype("float"))
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    _TIMEDELTA_DATA,
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_dt_properties(data, dtype):
    gsr = cudf.Series(data, dtype=dtype)
    psr = gsr.to_pandas()

    def local_assert(expected, actual):
        if gsr.isnull().any():
            assert_eq(expected, actual.astype("float"))
        else:
            assert_eq(expected, actual)

    expected_days = psr.dt.days
    actual_days = gsr.dt.days

    local_assert(expected_days, actual_days)

    expected_seconds = psr.dt.seconds
    actual_seconds = gsr.dt.seconds

    local_assert(expected_seconds, actual_seconds)

    expected_microseconds = psr.dt.microseconds
    actual_microseconds = gsr.dt.microseconds

    local_assert(expected_microseconds, actual_microseconds)

    expected_nanoseconds = psr.dt.nanoseconds
    actual_nanoseconds = gsr.dt.nanoseconds

    local_assert(expected_nanoseconds, actual_nanoseconds)


@pytest.mark.parametrize(
    "data",
    _TIMEDELTA_DATA,
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_index(data, dtype):
    gdi = cudf.Index(data, dtype=dtype)
    pdi = gdi.to_pandas()

    assert_eq(pdi, gdi)


@pytest.mark.parametrize("data", _TIMEDELTA_DATA_NON_OVERFLOW)
@pytest.mark.parametrize("datetime_dtype", utils.DATETIME_TYPES)
@pytest.mark.parametrize("timedelta_dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_index_datetime_index_ops(
    data, datetime_dtype, timedelta_dtype
):
    gdt = cudf.Index(data, dtype=datetime_dtype)
    gtd = cudf.Index(data, dtype=timedelta_dtype)

    pdt = gdt.to_pandas()
    ptd = gtd.to_pandas()

    assert_eq(gdt - gtd, pdt - ptd)
    assert_eq(gdt + gtd, pdt + ptd)


@pytest.mark.parametrize(
    "datetime_data,timedelta_data",
    [
        ([1000000, 200000, 3000000], [1000000, 200000, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, None]),
        ([], []),
        ([None], [None]),
        ([None, None, None, None, None], [None, None, None, None, None]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [12, 12, 22, 343, 4353534, 435342],
        ),
        (np.array([10, 20, 30, None, 100]), np.array([10, 20, 30, None, 100])),
        (cp.asarray([10, 20, 30, 100]), cp.asarray([10, 20, 30, 100])),
        ([1000000, 200000, 3000000], [200000, 34543, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, 3000000]),
        ([None], [1]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [None, 1, 220, 3, 34, 4353423287],
        ),
        (np.array([10, 20, 30, None, 100]), np.array([10, 20, 30, None, 100])),
        (cp.asarray([10, 20, 30, 100]), cp.asarray([10, 20, 30, 100])),
        (
            [12, 11, 232, 223432411, 2343241, 234324, 23234],
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
        ),
        (
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
        ),
        (
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        ),
        (
            [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        ),
    ],
)
@pytest.mark.parametrize("datetime_dtype", utils.DATETIME_TYPES)
@pytest.mark.parametrize("timedelta_dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_datetime_index_ops_misc(
    datetime_data, timedelta_data, datetime_dtype, timedelta_dtype
):
    gdt = cudf.Index(datetime_data, dtype=datetime_dtype)
    gtd = cudf.Index(timedelta_data, dtype=timedelta_dtype)

    pdt = gdt.to_pandas()
    ptd = gtd.to_pandas()

    assert_eq(gdt - gtd, pdt - ptd)
    assert_eq(gdt + gtd, pdt + ptd)


@pytest.mark.parametrize("data", _TIMEDELTA_DATA_NON_OVERFLOW)
@pytest.mark.parametrize(
    "other_scalars",
    [
        pd.Timedelta(1513393355.5, unit="s"),
        pd.Timedelta(34765, unit="D"),
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        datetime.timedelta(minutes=447),
        datetime.timedelta(hours=447),
        datetime.timedelta(weeks=734),
        np.timedelta64(4, "s"),
        np.timedelta64(456, "D"),
        np.timedelta64(46, "h"),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize(
    "op",
    [
        "add",
        "sub",
        "truediv",
        pytest.param(
            "floordiv",
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/35529"
            ),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning:pandas")
def test_timedelta_index_ops_with_scalars(data, other_scalars, dtype, op):
    gtdi = cudf.Index(data=data, dtype=dtype)
    ptdi = gtdi.to_pandas()

    if op == "add":
        expected = ptdi + other_scalars
        actual = gtdi + other_scalars
    elif op == "sub":
        expected = ptdi - other_scalars
        actual = gtdi - other_scalars
    elif op == "truediv":
        expected = ptdi / other_scalars
        actual = gtdi / other_scalars
    elif op == "floordiv":
        expected = ptdi // other_scalars
        actual = gtdi // other_scalars

    assert_eq(expected, actual)

    if op == "add":
        expected = other_scalars + ptdi
        actual = other_scalars + gtdi
    elif op == "sub":
        expected = other_scalars - ptdi
        actual = other_scalars - gtdi
    elif op == "truediv":
        expected = other_scalars / ptdi
        actual = other_scalars / gtdi
    elif op == "floordiv":
        expected = other_scalars // ptdi
        actual = other_scalars // gtdi

    assert_eq(expected, actual)


@pytest.mark.parametrize("data", _TIMEDELTA_DATA_NON_OVERFLOW)
@pytest.mark.parametrize(
    "cpu_scalar",
    [
        pd.Timedelta(1513393355.5, unit="s"),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize(
    "op",
    [
        "add",
        "sub",
        "truediv",
        pytest.param(
            "floordiv",
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/5938"
            ),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning:pandas")
def test_timedelta_index_ops_with_cudf_scalars(data, cpu_scalar, dtype, op):
    gtdi = cudf.Index(data=data, dtype=dtype)
    ptdi = gtdi.to_pandas()

    gpu_scalar = cudf.Scalar(cpu_scalar)

    if op == "add":
        expected = ptdi + cpu_scalar
        actual = gtdi + gpu_scalar
    elif op == "sub":
        expected = ptdi - cpu_scalar
        actual = gtdi - gpu_scalar
    elif op == "truediv":
        expected = ptdi / cpu_scalar
        actual = gtdi / gpu_scalar
    elif op == "floordiv":
        expected = ptdi // cpu_scalar
        actual = gtdi // gpu_scalar

    assert_eq(expected, actual)

    if op == "add":
        expected = cpu_scalar + ptdi
        actual = gpu_scalar + gtdi
    elif op == "sub":
        expected = cpu_scalar - ptdi
        actual = gpu_scalar - gtdi
    elif op == "truediv":
        expected = cpu_scalar / ptdi
        actual = gpu_scalar / gtdi
    elif op == "floordiv":
        expected = cpu_scalar // ptdi
        actual = gpu_scalar // gtdi

    assert_eq(expected, actual)


@pytest.mark.parametrize("data", _TIMEDELTA_DATA)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize("name", ["abcd", None])
def test_timedelta_index_properties(data, dtype, name):
    gdi = cudf.Index(data, dtype=dtype, name=name)
    pdi = gdi.to_pandas()

    def local_assert(expected, actual):
        if actual._values.null_count:
            assert_eq(expected, actual.astype("float64"))
        else:
            assert_eq(expected, actual)

    expected_days = pdi.days
    actual_days = gdi.days

    local_assert(expected_days, actual_days)

    expected_seconds = pdi.seconds
    actual_seconds = gdi.seconds

    local_assert(expected_seconds, actual_seconds)

    expected_microseconds = pdi.microseconds
    actual_microseconds = gdi.microseconds

    local_assert(expected_microseconds, actual_microseconds)

    expected_nanoseconds = pdi.nanoseconds
    actual_nanoseconds = gdi.nanoseconds

    local_assert(expected_nanoseconds, actual_nanoseconds)

    expected_components = pdi.components
    actual_components = gdi.components

    if actual_components.isnull().any().any():
        assert_eq(expected_components, actual_components.astype("float"))
    else:
        assert_eq(
            expected_components,
            actual_components,
            check_index_type=not actual_components.empty,
        )


@pytest.mark.parametrize("data", _TIMEDELTA_DATA)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize(
    "fill_value",
    [
        np.timedelta64(4, "s"),
        np.timedelta64(456, "D"),
        np.timedelta64(46, "h"),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
def test_timedelta_fillna(data, dtype, fill_value):
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    expected = psr.dropna()
    actual = sr.dropna()

    assert_eq(expected, actual)

    expected = psr.fillna(fill_value)
    actual = sr.fillna(fill_value)
    assert_eq(expected, actual)

    expected = expected.dropna()
    actual = actual.dropna()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "gsr,expected_series",
    [
        (
            cudf.Series([1, 2, 3], dtype="timedelta64[ns]"),
            cudf.Series(
                [
                    "0 days 00:00:00.000000001",
                    "0 days 00:00:00.000000002",
                    "0 days 00:00:00.000000003",
                ]
            ),
        ),
        (
            cudf.Series([1000000, 200000, 3000000], dtype="timedelta64[ms]"),
            cudf.Series(
                ["0 days 00:16:40", "0 days 00:03:20", "0 days 00:50:00"]
            ),
        ),
        (
            cudf.Series([1000000, 200000, 3000000], dtype="timedelta64[s]"),
            cudf.Series(
                ["11 days 13:46:40", "2 days 07:33:20", "34 days 17:20:00"]
            ),
        ),
        (
            cudf.Series(
                [None, None, None, None, None], dtype="timedelta64[us]"
            ),
            cudf.Series([None, None, None, None, None], dtype="str"),
        ),
        (
            cudf.Series(
                [
                    136457654,
                    None,
                    245345345,
                    223432411,
                    None,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[us]",
            ),
            cudf.Series(
                [
                    "0 days 00:02:16.457654",
                    None,
                    "0 days 00:04:05.345345",
                    "0 days 00:03:43.432411",
                    None,
                    "0 days 01:00:34.548734",
                    "0 days 00:00:00.023234",
                ]
            ),
        ),
        (
            cudf.Series(
                [
                    136457654,
                    None,
                    245345345,
                    223432411,
                    None,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[ms]",
            ),
            cudf.Series(
                [
                    "1 days 13:54:17.654",
                    None,
                    "2 days 20:09:05.345",
                    "2 days 14:03:52.411",
                    None,
                    "42 days 01:35:48.734",
                    "0 days 00:00:23.234",
                ]
            ),
        ),
        (
            cudf.Series(
                [
                    136457654,
                    None,
                    245345345,
                    223432411,
                    None,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[s]",
            ),
            cudf.Series(
                [
                    "1579 days 08:54:14",
                    None,
                    "2839 days 15:29:05",
                    "2586 days 00:33:31",
                    None,
                    "42066 days 12:52:14",
                    "0 days 06:27:14",
                ]
            ),
        ),
        (
            cudf.Series(
                [
                    136457654,
                    None,
                    245345345,
                    223432411,
                    None,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[ns]",
            ),
            cudf.Series(
                [
                    "0 days 00:00:00.136457654",
                    None,
                    "0 days 00:00:00.245345345",
                    "0 days 00:00:00.223432411",
                    None,
                    "0 days 00:00:03.634548734",
                    "0 days 00:00:00.000023234",
                ]
            ),
        ),
    ],
)
def test_timedelta_str_roundtrip(gsr, expected_series):

    actual_series = gsr.astype("str")

    assert_eq(expected_series, actual_series)

    assert_eq(gsr, actual_series.astype(gsr.dtype))


def test_timedelta_invalid_ops():
    sr = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, "a"],),
        rfunc_args_and_kwargs=([sr, "a"],),
        compare_error_message=False,
    )

    dt_sr = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    dt_psr = dt_sr.to_pandas()

    assert_exceptions_equal(
        lfunc=operator.mod,
        rfunc=operator.mod,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.mod,
        rfunc=operator.mod,
        lfunc_args_and_kwargs=([psr, "a"],),
        rfunc_args_and_kwargs=([sr, "a"],),
        check_exception_type=False,
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.gt,
        rfunc=operator.gt,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.lt,
        rfunc=operator.lt,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.ge,
        rfunc=operator.ge,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.le,
        rfunc=operator.le,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.mul,
        rfunc=operator.mul,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.mul,
        rfunc=operator.mul,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
        check_exception_type=False,
        compare_error_message=False,
    )

    assert_exceptions_equal(
        lfunc=operator.xor,
        rfunc=operator.xor,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
        compare_error_message=False,
    )


def test_timedelta_datetime_cast_invalid():
    sr = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        psr.astype,
        sr.astype,
        (["datetime64[ns]"],),
        (["datetime64[ns]"],),
        expected_error_message=re.escape(
            "cannot astype a timedelta from timedelta64[ns] to datetime64[ns]"
        ),
    )

    sr = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        psr.astype,
        sr.astype,
        (["timedelta64[ns]"],),
        (["timedelta64[ns]"],),
        expected_error_message=re.escape(
            "cannot astype a datetimelike from "
            "datetime64[ns] to timedelta64[ns]"
        ),
    )


@pytest.mark.parametrize("data", [[], [1, 2, 3, 4, 5]])
@pytest.mark.parametrize("dtype", utils.NUMERIC_TYPES)
@pytest.mark.parametrize("timedelta_dtype", utils.TIMEDELTA_TYPES)
def test_numeric_to_timedelta(data, dtype, timedelta_dtype):
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    actual = sr.astype(timedelta_dtype)
    expected = pd.Series(psr.to_numpy().astype(timedelta_dtype))

    assert_eq(expected, actual)


@pytest.mark.parametrize("data", [[], [1, 2, 3, 4, 5]])
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize(
    "scalar",
    [
        1,
        2,
        3,
        "a",
        np.timedelta64(1, "s"),
        np.timedelta64(2, "s"),
        np.timedelta64(2, "D"),
        np.timedelta64(3, "ms"),
        np.timedelta64(4, "us"),
        np.timedelta64(5, "ns"),
        np.timedelta64(6, "ns"),
        np.datetime64(6, "s"),
    ],
)
def test_timedelta_contains(data, dtype, scalar):
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    expected = scalar in sr
    actual = scalar in psr

    assert_eq(expected, actual)


@pytest.mark.parametrize("data", [[1, 2, 3], [], [1, 20, 1000, None]])
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize("ddof", [1, 2, 3])
def test_timedelta_std(data, dtype, ddof):
    gsr = cudf.Series(data, dtype=dtype)
    psr = gsr.to_pandas()

    expected = psr.std(ddof=ddof)
    actual = gsr.std(ddof=ddof)

    if np.isnat(expected.to_numpy()) and np.isnat(actual.to_numpy()):
        assert True
    else:
        np.testing.assert_allclose(
            expected.to_numpy().astype("float64"),
            actual.to_numpy().astype("float64"),
            rtol=1e-5,
            atol=0,
        )


@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 100],
        [10, None, 100, None, None],
        [None, None, None],
        [1231],
    ],
)
@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
def test_timedelta_reductions(data, op, dtype):
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    actual = getattr(sr, op)()
    expected = getattr(psr, op)()

    if np.isnat(expected.to_numpy()) and np.isnat(actual):
        assert True
    else:
        assert_eq(expected.to_numpy(), actual)


def test_error_values():
    s = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    with pytest.raises(
        NotImplementedError,
        match="TimeDelta Arrays is not yet implemented in cudf",
    ):
        s.values


@pytest.mark.parametrize("dtype", utils.TIMEDELTA_TYPES)
@pytest.mark.parametrize("name", [None, "delta-index"])
def test_create_TimedeltaIndex(dtype, name):
    gdi = cudf.TimedeltaIndex(
        [1132223, 2023232, 342234324, 4234324], dtype=dtype, name=name
    )
    pdi = gdi.to_pandas()
    assert_eq(pdi, gdi)


@pytest.mark.parametrize("data", [[43534, 43543, 37897, 2000]])
@pytest.mark.parametrize("dtype", ["timedelta64[ns]"])
def test_timedelta_constructor(data, dtype):
    expected = pd.TimedeltaIndex(data=data, dtype=dtype)
    actual = cudf.TimedeltaIndex(data=data, dtype=dtype)

    assert_eq(expected, actual)

    expected = pd.TimedeltaIndex(data=pd.Series(data), dtype=dtype)
    actual = cudf.TimedeltaIndex(data=cudf.Series(data), dtype=dtype)

    assert_eq(expected, actual)
