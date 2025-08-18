# Copyright (c) 2025, NVIDIA CORPORATION.
import datetime
import operator

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "sr1", [pd.Series([10, 11, 12], index=["a", "b", "z"]), pd.Series(["a"])]
)
@pytest.mark.parametrize(
    "sr2",
    [pd.Series([], dtype="float64"), pd.Series(["a", "a", "c", "z", "A"])],
)
def test_series_error_equality(sr1, sr2, comparison_op):
    gsr1 = cudf.from_pandas(sr1)
    gsr2 = cudf.from_pandas(sr2)

    assert_exceptions_equal(
        comparison_op, comparison_op, ([sr1, sr2],), ([gsr1, gsr2],)
    )


@pytest.mark.parametrize(
    "data,other",
    [
        ([1000000, 200000, 3000000], [1000000, 200000, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, None]),
        ([], []),
        ([None], [None]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [12, 12, 22, 343, 4353534, 435342],
        ),
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
def test_timedelta_ops_misc_inputs(
    data, other, timedelta_types_as_str, binary_op_method
):
    if binary_op_method in {"mul", "rmul", "pow", "rpow"}:
        pytest.skip(f"Test not applicable for {binary_op_method}")
    gsr = cudf.Series(data, dtype=timedelta_types_as_str)
    other_gsr = cudf.Series(other, dtype=timedelta_types_as_str)

    psr = gsr.to_pandas()
    other_psr = other_gsr.to_pandas()

    expected = getattr(psr, binary_op_method)(other_psr)
    actual = getattr(gsr, binary_op_method)(other_gsr)
    if binary_op_method in ("eq", "lt", "gt", "le", "ge"):
        actual = actual.fillna(False)
    elif binary_op_method == "ne":
        actual = actual.fillna(True)

    if binary_op_method == "floordiv":
        expected[actual.isna().to_pandas()] = np.nan

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "datetime_data,timedelta_data",
    [
        ([1000000, 200000, 3000000], [1000000, 200000, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, None]),
        ([], []),
        ([None], [None]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [12, 12, 22, 343, 4353534, 435342],
        ),
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
@pytest.mark.parametrize("ops", ["add", "sub"])
def test_timedelta_ops_datetime_inputs(
    datetime_types_as_str,
    timedelta_types_as_str,
    datetime_data,
    timedelta_data,
    ops,
):
    gsr_datetime = cudf.Series(datetime_data, dtype=datetime_types_as_str)
    gsr_timedelta = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)

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
                "B": pd.Series(
                    pd.timedelta_range(start="1 day", periods=3, freq="D")
                ),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series(
                    pd.date_range("1994-1-1", periods=10, freq="D")
                ),
                "B": pd.Series(
                    pd.timedelta_range(start="1 day", periods=10, freq="D")
                ),
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
def test_timedelta_series_ops_with_scalars(
    data, other_scalars, timedelta_types_as_str, arithmetic_op_method, request
):
    if arithmetic_op_method in {
        "mul",
        "rmul",
        "rtruediv",
        "pow",
        "rpow",
        "radd",
        "rsub",
        "rfloordiv",
        "rmod",
    }:
        pytest.skip(f"Test not applicable for {arithmetic_op_method}")
    gsr = cudf.Series(data=data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    if arithmetic_op_method == "add":
        expected = psr + other_scalars
        actual = gsr + other_scalars
    elif arithmetic_op_method == "sub":
        expected = psr - other_scalars
        actual = gsr - other_scalars
    elif arithmetic_op_method == "truediv":
        expected = psr / other_scalars
        actual = gsr / other_scalars
    elif arithmetic_op_method == "floordiv":
        expected = psr // other_scalars
        actual = gsr // other_scalars
    elif arithmetic_op_method == "mod":
        expected = psr % other_scalars
        actual = gsr % other_scalars

    assert_eq(expected, actual)

    if arithmetic_op_method == "add":
        expected = other_scalars + psr
        actual = other_scalars + gsr
    elif arithmetic_op_method == "sub":
        expected = other_scalars - psr
        actual = other_scalars - gsr
    elif arithmetic_op_method == "truediv":
        expected = other_scalars / psr
        actual = other_scalars / gsr
    elif arithmetic_op_method == "floordiv":
        expected = other_scalars // psr
        actual = other_scalars // gsr
    elif arithmetic_op_method == "mod":
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


@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_timdelta_binop_tz_timestamp(op):
    s = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    pd_tz_timestamp = pd.Timestamp("1970-01-01 00:00:00.000000001", tz="utc")
    with pytest.raises(NotImplementedError):
        op(s, pd_tz_timestamp)
    date_tz_scalar = datetime.datetime.now(datetime.timezone.utc)
    with pytest.raises(NotImplementedError):
        op(s, date_tz_scalar)


def test_timedelta_series_cmpops_pandas_compatibility(comparison_op):
    gsr1 = cudf.Series(
        data=[123, 456, None, 321, None], dtype="timedelta64[ns]"
    )
    psr1 = gsr1.to_pandas()

    gsr2 = cudf.Series(
        data=[123, 456, 789, None, None], dtype="timedelta64[ns]"
    )
    psr2 = gsr2.to_pandas()

    expect = comparison_op(psr1, psr2)
    with cudf.option_context("mode.pandas_compatible", True):
        got = comparison_op(gsr1, gsr2)

    assert_eq(expect, got)


def test_string_equality():
    data1 = ["b", "c", "d", "a", "c"]
    data2 = ["a", None, "c", "a", "c"]

    ps1 = pd.Series(data1)
    ps2 = pd.Series(data2)
    gs1 = cudf.Series(data1)
    gs2 = cudf.Series(data2)

    expect = ps1 == ps2
    got = gs1 == gs2

    assert_eq(expect, got.fillna(False))

    expect = ps1 == "m"
    got = gs1 == "m"

    assert_eq(expect, got.fillna(False))

    ps1 = pd.Series(["a"])
    gs1 = cudf.Series(["a"])

    expect = ps1 == "m"
    got = gs1 == "m"

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "lhs",
    [
        ["Cbe", "cbe", "CbeD", "Cb", "ghi", "Cb"],
        ["abc", "xyz", "a", "ab", "123", "097"],
    ],
)
@pytest.mark.parametrize(
    "rhs",
    [
        ["Cbe", "cbe", "CbeD", "Cb", "ghi", "Cb"],
        ["a", "a", "a", "a", "A", "z"],
    ],
)
def test_string_binary_op_add(lhs, rhs):
    pds = pd.Series(lhs) + pd.Series(rhs)
    gds = cudf.Series(lhs) + cudf.Series(rhs)

    assert_eq(pds, gds)
