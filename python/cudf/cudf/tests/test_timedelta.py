# Copyright (c) 2020-2025, NVIDIA CORPORATION.

import datetime
import operator

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_230
from cudf.testing import _utils as utils, assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@pytest.fixture(
    params=[
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
)
def data_non_overflow(request):
    return request.param


@pytest.fixture(params=utils.TIMEDELTA_TYPES)
def timedelta_dtype(request):
    return request.param


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
def test_timedelta_ops_misc_inputs(data, other, timedelta_dtype, ops):
    gsr = cudf.Series(data, dtype=timedelta_dtype)
    other_gsr = cudf.Series(other, dtype=timedelta_dtype)

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
        [1.321, 1132.324, 23223231.11, 233.41, 332, 323],
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
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        "add",
        "sub",
        "truediv",
        "mod",
        "floordiv",
    ],
)
def test_timedelta_series_ops_with_scalars(
    data, other_scalars, timedelta_dtype, op
):
    gsr = cudf.Series(data=data, dtype=timedelta_dtype)
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
    "reverse",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "timedelta modulo by zero is dubiously defined in "
                    "both pandas and cuDF "
                    "(see https://github.com/rapidsai/cudf/issues/5938)"
                ),
            ),
        ),
    ],
)
def test_timedelta_series_mod_with_scalar_zero(reverse):
    gsr = cudf.Series(data=[0.2434], dtype=np.timedelta64(1, "ns"))
    psr = gsr.to_pandas()
    scalar = datetime.timedelta(days=768)
    if reverse:
        expected = scalar % psr
        actual = scalar % gsr
    else:
        expected = psr % scalar
        actual = gsr % scalar
    assert_eq(expected, actual)


@pytest.mark.parametrize("reduction_op", ["sum", "mean", "median", "quantile"])
def test_timedelta_reduction_ops(
    data_non_overflow, timedelta_dtype, reduction_op
):
    gsr = cudf.Series(data_non_overflow, dtype=timedelta_dtype)
    psr = gsr.to_pandas()

    if len(psr) > 0 and psr.isnull().all() and reduction_op == "median":
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            expected = getattr(psr, reduction_op)()
    else:
        with expect_warning_if(
            PANDAS_GE_230
            and reduction_op == "quantile"
            and len(data_non_overflow) == 0
            and timedelta_dtype != "timedelta64[ns]"
        ):
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


@pytest.mark.parametrize("datetime_dtype", utils.DATETIME_TYPES)
def test_timedelta_index_datetime_index_ops(
    data_non_overflow, datetime_dtype, timedelta_dtype
):
    gdt = cudf.Index(data_non_overflow, dtype=datetime_dtype)
    gtd = cudf.Index(data_non_overflow, dtype=timedelta_dtype)

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
def test_timedelta_datetime_index_ops_misc(
    datetime_data, timedelta_data, datetime_dtype, timedelta_dtype
):
    gdt = cudf.Index(datetime_data, dtype=datetime_dtype)
    gtd = cudf.Index(timedelta_data, dtype=timedelta_dtype)

    pdt = gdt.to_pandas()
    ptd = gtd.to_pandas()

    assert_eq(gdt - gtd, pdt - ptd)
    assert_eq(gdt + gtd, pdt + ptd)


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
@pytest.mark.parametrize(
    "op",
    [
        "add",
        "sub",
        "truediv",
        "floordiv",
    ],
)
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning:pandas")
def test_timedelta_index_ops_with_scalars(
    request, data_non_overflow, other_scalars, timedelta_dtype, op
):
    gtdi = cudf.Index(data=data_non_overflow, dtype=timedelta_dtype)
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

    # Division by zero for datetime or timedelta is
    # dubiously defined in both pandas (Any // 0 -> 0 in
    # pandas) and cuDF (undefined behaviour)
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                op == "floordiv"
                and 0 in ptdi.astype("int")
                and np.timedelta64(other_scalars).item() is not None
            ),
            reason="Related to https://github.com/rapidsai/cudf/issues/5938",
        )
    )
    assert_eq(expected, actual)


def test_timedelta_invalid_ops():
    sr = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, "a"],),
        rfunc_args_and_kwargs=([sr, "a"],),
    )

    dt_sr = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    dt_psr = dt_sr.to_pandas()

    assert_exceptions_equal(
        lfunc=operator.mod,
        rfunc=operator.mod,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.mod,
        rfunc=operator.mod,
        lfunc_args_and_kwargs=([psr, "a"],),
        rfunc_args_and_kwargs=([sr, "a"],),
        check_exception_type=False,
    )

    assert_exceptions_equal(
        lfunc=operator.gt,
        rfunc=operator.gt,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.lt,
        rfunc=operator.lt,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.ge,
        rfunc=operator.ge,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.le,
        rfunc=operator.le,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.mul,
        rfunc=operator.mul,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.mul,
        rfunc=operator.mul,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
        check_exception_type=False,
    )

    assert_exceptions_equal(
        lfunc=operator.xor,
        rfunc=operator.xor,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
    )


@pytest.mark.parametrize("data", [[1, 2, 3], [], [1, 20, 1000, None]])
@pytest.mark.parametrize("ddof", [1, 2, 3])
def test_timedelta_std(data, timedelta_dtype, ddof):
    gsr = cudf.Series(data, dtype=timedelta_dtype)
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
def test_timedelta_reductions(data, op, timedelta_dtype):
    sr = cudf.Series(data, dtype=timedelta_dtype)
    psr = sr.to_pandas()

    actual = getattr(sr, op)()
    expected = getattr(psr, op)()

    if np.isnat(expected.to_numpy()) and np.isnat(actual):
        assert True
    else:
        assert_eq(expected.to_numpy(), actual)


@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_timdelta_binop_tz_timestamp(op):
    s = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    pd_tz_timestamp = pd.Timestamp("1970-01-01 00:00:00.000000001", tz="utc")
    with pytest.raises(NotImplementedError):
        op(s, pd_tz_timestamp)
    date_tz_scalar = datetime.datetime.now(datetime.timezone.utc)
    with pytest.raises(NotImplementedError):
        op(s, date_tz_scalar)


@pytest.mark.parametrize(
    "op",
    [
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
        operator.eq,
        operator.ne,
    ],
)
def test_timedelta_series_cmpops_pandas_compatibility(op):
    gsr1 = cudf.Series(
        data=[123, 456, None, 321, None], dtype="timedelta64[ns]"
    )
    psr1 = gsr1.to_pandas()

    gsr2 = cudf.Series(
        data=[123, 456, 789, None, None], dtype="timedelta64[ns]"
    )
    psr2 = gsr2.to_pandas()

    expect = op(psr1, psr2)
    with cudf.option_context("mode.pandas_compatible", True):
        got = op(gsr1, gsr2)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ["sum", {}],
        ["mean", {}],
        ["median", {}],
        ["std", {}],
        ["std", {"ddof": 0}],
    ],
)
def test_tdi_reductions(method, kwargs):
    pd_tdi = pd.TimedeltaIndex(["1 day", "2 days", "3 days"])
    cudf_tdi = cudf.from_pandas(pd_tdi)

    result = getattr(pd_tdi, method)(**kwargs)
    expected = getattr(cudf_tdi, method)(**kwargs)
    assert result == expected
