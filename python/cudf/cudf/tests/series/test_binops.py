# Copyright (c) 2025, NVIDIA CORPORATION.
import datetime
import decimal
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


def test_concatenate_rows_of_lists():
    pser = pd.Series([["a", "a"], ["b"], ["c"]])
    gser = cudf.Series([["a", "a"], ["b"], ["c"]])

    expect = pser + pser
    got = gser + gser

    assert_eq(expect, got)


def test_concatenate_list_with_nonlist():
    gser1 = cudf.Series([["a", "c"], ["b", "d"], ["c", "d"]])
    gser2 = cudf.Series(["a", "b", "c"])
    with pytest.raises(TypeError):
        gser1 + gser2


def test_datetime_series_binops_pandas(
    datetime_types_as_str, datetime_types_as_str2
):
    dti = pd.date_range("20010101", "20020215", freq="400h", name="times")
    pd_data_1 = pd.Series(dti)
    pd_data_2 = pd_data_1
    gdf_data_1 = cudf.Series(pd_data_1).astype(datetime_types_as_str)
    gdf_data_2 = cudf.Series(pd_data_2).astype(datetime_types_as_str2)
    assert_eq(pd_data_1, gdf_data_1.astype("datetime64[ns]"))
    assert_eq(pd_data_2, gdf_data_2.astype("datetime64[ns]"))
    assert_eq(pd_data_1 < pd_data_2, gdf_data_1 < gdf_data_2)
    assert_eq(pd_data_1 > pd_data_2, gdf_data_1 > gdf_data_2)
    assert_eq(pd_data_1 == pd_data_2, gdf_data_1 == gdf_data_2)
    assert_eq(pd_data_1 <= pd_data_2, gdf_data_1 <= gdf_data_2)
    assert_eq(pd_data_1 >= pd_data_2, gdf_data_1 >= gdf_data_2)


def test_datetime_series_binops_numpy(
    datetime_types_as_str, datetime_types_as_str2
):
    dti = pd.date_range("20010101", "20020215", freq="400h", name="times")
    pd_data_1 = pd.Series(dti)
    pd_data_2 = pd_data_1
    gdf_data_1 = cudf.Series(pd_data_1).astype(datetime_types_as_str)
    gdf_data_2 = cudf.Series(pd_data_2).astype(datetime_types_as_str2)
    np_data_1 = np.array(pd_data_1).astype(datetime_types_as_str)
    np_data_2 = np.array(pd_data_2).astype(datetime_types_as_str2)
    np.testing.assert_equal(np_data_1, gdf_data_1.to_numpy())
    np.testing.assert_equal(np_data_2, gdf_data_2.to_numpy())
    np.testing.assert_equal(
        np.less(np_data_1, np_data_2), (gdf_data_1 < gdf_data_2).to_numpy()
    )
    np.testing.assert_equal(
        np.greater(np_data_1, np_data_2), (gdf_data_1 > gdf_data_2).to_numpy()
    )
    np.testing.assert_equal(
        np.equal(np_data_1, np_data_2), (gdf_data_1 == gdf_data_2).to_numpy()
    )
    np.testing.assert_equal(
        np.less_equal(np_data_1, np_data_2),
        (gdf_data_1 <= gdf_data_2).to_numpy(),
    )
    np.testing.assert_equal(
        np.greater_equal(np_data_1, np_data_2),
        (gdf_data_1 >= gdf_data_2).to_numpy(),
    )


@pytest.mark.parametrize(
    "data",
    [
        pd.date_range("20010101", "20020215", freq="400h", name="times"),
        pd.date_range(
            "20010101", freq="243434324423423234ns", name="times", periods=10
        ),
    ],
)
def test_dt_ops(data):
    pd_data = pd.Series(data)
    gdf_data = cudf.Series(data)

    assert_eq(pd_data == pd_data, gdf_data == gdf_data)
    assert_eq(pd_data < pd_data, gdf_data < gdf_data)
    assert_eq(pd_data > pd_data, gdf_data > gdf_data)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 10, 100, 20000],
        [None] * 7,
        [10, 20, 30, None, 100, 200, None],
        [3223.234, 342.2332, 23423.23, 3343.23324, 23432.2323, 242.23, 233],
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [1, 2, 3, 4, 10, 100, 20000],
        [None] * 7,
        [10, 20, 30, None, 100, 200, None],
        [3223.234, 342.2332, 23423.23, 3343.23324, 23432.2323, 242.23, 233],
        datetime.datetime(1993, 6, 22, 13, 30),
        datetime.datetime(2005, 1, 22, 10, 00),
        np.datetime64("2005-02"),
        np.datetime64("2005-02-25"),
        np.datetime64("2005-02-25T03:30"),
        np.datetime64("nat"),
        # TODO: https://github.com/pandas-dev/pandas/issues/52295
    ],
)
def test_datetime_subtract(
    data, other, datetime_types_as_str, datetime_types_as_str2
):
    gsr = cudf.Series(data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    if isinstance(other, np.datetime64):
        gsr_other = other
        psr_other = other
    else:
        gsr_other = cudf.Series(other, dtype=datetime_types_as_str2)
        psr_other = gsr_other.to_pandas()

    expected = psr - psr_other
    actual = gsr - gsr_other

    assert_eq(expected, actual)

    expected = psr_other - psr
    actual = gsr_other - gsr

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
@pytest.mark.parametrize("op", ["add", "sub"])
def test_datetime_series_ops_with_scalars(
    data, other_scalars, datetime_types_as_str, op
):
    gsr = cudf.Series(data=data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    if op == "add":
        expected = psr + other_scalars
        actual = gsr + other_scalars
    elif op == "sub":
        expected = psr - other_scalars
        actual = gsr - other_scalars

    assert_eq(expected, actual)

    if op == "add":
        expected = other_scalars + psr
        actual = other_scalars + gsr

        assert_eq(expected, actual)

    elif op == "sub":
        assert_exceptions_equal(
            lfunc=operator.sub,
            rfunc=operator.sub,
            lfunc_args_and_kwargs=([other_scalars, psr],),
            rfunc_args_and_kwargs=([other_scalars, gsr],),
        )


@pytest.mark.parametrize("data", ["20110101", "20120101", "20130101"])
@pytest.mark.parametrize("other_scalars", ["20110101", "20120101", "20130101"])
def test_datetime_series_cmpops_with_scalars(
    data, other_scalars, datetime_types_as_str, comparison_op
):
    gsr = cudf.Series(data=data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    expect = comparison_op(psr, other_scalars)
    got = comparison_op(gsr, other_scalars)

    assert_eq(expect, got)


def test_datetime_invalid_ops():
    sr = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, "a"],),
        rfunc_args_and_kwargs=([sr, "a"],),
    )

    assert_exceptions_equal(
        lfunc=operator.mul,
        rfunc=operator.mul,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
    )


def test_datetime_binop_tz_timestamp(comparison_op):
    s = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    pd_tz_timestamp = pd.Timestamp("1970-01-01 00:00:00.000000001", tz="utc")
    with pytest.raises(NotImplementedError):
        comparison_op(s, pd_tz_timestamp)

    date_scalar = datetime.datetime.now(datetime.timezone.utc)
    with pytest.raises(NotImplementedError):
        comparison_op(s, date_scalar)


def test_datetime_series_cmpops_pandas_compatibility(comparison_op):
    data1 = ["20110101", "20120101", None, "20140101", None]
    data2 = ["20110101", "20120101", "20130101", None, None]
    gsr1 = cudf.Series(data=data1, dtype="datetime64[ns]")
    psr1 = gsr1.to_pandas()

    gsr2 = cudf.Series(data=data2, dtype="datetime64[ns]")
    psr2 = gsr2.to_pandas()

    expect = comparison_op(psr1, psr2)
    with cudf.option_context("mode.pandas_compatible", True):
        got = comparison_op(gsr1, gsr2)

    assert_eq(expect, got)


def test_decimal_overflow():
    s = cudf.Series(
        [decimal.Decimal("0.0009384233522166997927180531650178250")]
    )
    result = s * s
    assert_eq(cudf.Decimal128Dtype(precision=38, scale=37), result.dtype)

    s = cudf.Series([1, 2], dtype=cudf.Decimal128Dtype(precision=38, scale=0))
    result = s * decimal.Decimal("1.0")
    assert_eq(cudf.Decimal128Dtype(precision=38, scale=1), result.dtype)


def test_decimal_binop_upcast_operands():
    ser1 = cudf.Series([0.51, 1.51, 2.51]).astype(cudf.Decimal64Dtype(18, 2))
    ser2 = cudf.Series([0.90, 0.96, 0.99]).astype(cudf.Decimal128Dtype(19, 2))
    result = ser1 + ser2
    expected = cudf.Series([1.41, 2.47, 3.50]).astype(
        cudf.Decimal128Dtype(20, 2)
    )
    assert_eq(result, expected)


def test_categorical_compare_ordered():
    cat1 = pd.Categorical(
        ["a", "a", "b", "c", "a"], categories=["a", "b", "c"], ordered=True
    )
    pdsr1 = pd.Series(cat1)
    sr1 = cudf.Series(cat1)
    cat2 = pd.Categorical(
        ["a", "b", "a", "c", "b"], categories=["a", "b", "c"], ordered=True
    )
    pdsr2 = pd.Series(cat2)
    sr2 = cudf.Series(cat2)

    # test equal
    out = sr1 == sr1
    assert out.dtype == np.bool_
    assert type(out[0]) is np.bool_
    assert np.all(out.to_numpy())
    assert np.all(pdsr1 == pdsr1)

    # test inequality
    out = sr1 != sr1
    assert not np.any(out.to_numpy())
    assert not np.any(pdsr1 != pdsr1)

    assert pdsr1.cat.ordered
    assert sr1.cat.ordered

    # test using ordered operators
    np.testing.assert_array_equal(pdsr1 < pdsr2, (sr1 < sr2).to_numpy())
    np.testing.assert_array_equal(pdsr1 > pdsr2, (sr1 > sr2).to_numpy())


def test_categorical_binary_add():
    cat = pd.Categorical(["a", "a", "b", "c", "a"], categories=["a", "b", "c"])
    pdsr = pd.Series(cat)
    sr = cudf.Series(cat)

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([pdsr, pdsr],),
        rfunc_args_and_kwargs=([sr, sr],),
    )
