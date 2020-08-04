# Copyright (c) 2020, NVIDIA CORPORATION.
import datetime

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq
from cudf.utils import dtypes as dtypeutils


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
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
def test_timedelta_series_create(data, dtype):
    if dtype not in ("timedelta64[ns]"):
        pytest.skip(
            "Bug in pandas" "https://github.com/pandas-dev/pandas/issues/35465"
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
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
@pytest.mark.parametrize("cast_dtype", ["int64", "category", "object"])
def test_timedelta_from_typecast(data, dtype, cast_dtype):
    if dtype not in ("timedelta64[ns]"):
        pytest.skip(
            "Bug in pandas" "https://github.com/pandas-dev/pandas/issues/35465"
        )
    psr = pd.Series(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data, dtype=dtype
    )
    gsr = cudf.Series(data, dtype=dtype)

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
@pytest.mark.parametrize("cast_dtype", dtypeutils.TIMEDELTA_TYPES)
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
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
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
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
@pytest.mark.parametrize("fillna", [None, "pandas"])
def test_timedelta_series_to_array(data, dtype, fillna):
    gsr = cudf.Series(data, dtype=dtype)

    expected = np.array(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data, dtype=dtype
    )
    if fillna is None:
        expected = expected[~np.isnan(expected)]

    actual = gsr.to_array(fillna=fillna)

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
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
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
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
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
    ],
)
def test_timedelta_ops_misc_inputs(data, other, dtype, ops):
    gsr = cudf.Series(data, dtype=dtype)
    other_gsr = cudf.Series(other, dtype=dtype)

    psr = gsr.to_pandas(nullable_pd_dtype=True)
    other_psr = other_gsr.to_pandas(nullable_pd_dtype=True)

    expected = getattr(psr, ops)(other_psr)
    actual = getattr(gsr, ops)(other_gsr)
    if ops in ("eq", "lt", "gt", "le", "ge"):
        actual = actual.fillna(False)
    elif ops == "ne":
        actual = actual.fillna(True)

    if expected.dtype in cudf.utils.dtypes.cudf_dtypes_to_pandas_dtypes:
        expected = expected.astype(
            cudf.utils.dtypes.cudf_dtypes_to_pandas_dtypes[expected.dtype]
        )

    if ops == "floordiv":
        expected[actual.isna().to_pandas()] = pd.NA

    assert_eq(expected, actual, nullable_pd_dtype=True)


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
@pytest.mark.parametrize("datetime_dtype", dtypeutils.DATETIME_TYPES)
@pytest.mark.parametrize("timedelta_dtype", dtypeutils.TIMEDELTA_TYPES)
@pytest.mark.parametrize(
    "ops", ["add", "sub"],
)
def test_timedelta_ops_datetime_inputs(
    datetime_data, timedelta_data, datetime_dtype, timedelta_dtype, ops
):
    gsr_datetime = cudf.Series(datetime_data, dtype=datetime_dtype)
    gsr_timedelta = cudf.Series(timedelta_data, dtype=timedelta_dtype)

    psr_datetime = gsr_datetime.to_pandas(nullable_pd_dtype=True)
    psr_timedelta = gsr_timedelta.to_pandas(nullable_pd_dtype=True)

    expected = getattr(psr_datetime, ops)(psr_timedelta)
    actual = getattr(gsr_datetime, ops)(gsr_timedelta)

    assert_eq(expected, actual, nullable_pd_dtype=True)


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
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
@pytest.mark.parametrize("op", ["add", "sub", "truediv", "floordiv"])
def test_timedelta_series_ops_with_scalars(data, other_scalars, dtype, op):
    gsr = cudf.Series(data=data, dtype=dtype)
    psr = gsr.to_pandas(nullable_pd_dtype=True)

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
        # TODO: https://github.com/pandas-dev/pandas/issues/35529
        # uncomment this below line once above bug is fixed

        # expected = psr // other_scalars
        expected = (gsr / other_scalars).astype("int64")
        actual = gsr // other_scalars

    assert_eq(expected, actual, nullable_pd_dtype=True)

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
        # TODO: https://github.com/pandas-dev/pandas/issues/35529
        # uncomment this below line once above bug is fixed

        # expected = psr // other_scalars
        expected = (other_scalars / gsr).astype("int64")
        actual = other_scalars // gsr

    assert_eq(expected, actual, nullable_pd_dtype=True)


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
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
@pytest.mark.parametrize("reduction_op", ["sum", "mean", "median", "quantile"])
def test_timedelta_reduction_ops(data, dtype, reduction_op):
    gsr = cudf.Series(data, dtype=dtype)
    psr = gsr.to_pandas()

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
        print(expected, type(expected))
        print(actual, type(actual))
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
    ],
)
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
def test_timedelta_dt_components(data, dtype):
    gsr = cudf.Series(data, dtype=dtype)
    psr = gsr.to_pandas(nullable_pd_dtype=True)

    expected = psr.dt.components
    actual = gsr.dt.components

    if gsr.isnull().any():
        assert_eq(expected, actual.astype("float"))
    else:
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
    ],
)
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
def test_timedelta_dt_properties(data, dtype):
    gsr = cudf.Series(data, dtype=dtype)
    psr = gsr.to_pandas(nullable_pd_dtype=True)

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
    ],
)
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
def test_timedelta_index(data, dtype):
    gdi = cudf.Index(data, dtype=dtype)
    pdi = gdi.to_pandas()

    assert_eq(pdi, gdi)


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
        [13645765, 13473678, 24534534, 22343241, 2343241, 36345487, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize("datetime_dtype", dtypeutils.DATETIME_TYPES)
@pytest.mark.parametrize("timedelta_dtype", dtypeutils.TIMEDELTA_TYPES)
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
@pytest.mark.parametrize("datetime_dtype", dtypeutils.DATETIME_TYPES)
@pytest.mark.parametrize("timedelta_dtype", dtypeutils.TIMEDELTA_TYPES)
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
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
@pytest.mark.parametrize("op", ["add", "sub", "truediv", "floordiv"])
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
        # TODO: https://github.com/pandas-dev/pandas/issues/35529
        # uncomment this below line once above bug is fixed

        # expected = psr // other_scalars
        expected = (gtdi / other_scalars).astype("int64")
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
        # TODO: https://github.com/pandas-dev/pandas/issues/35529
        # uncomment this below line once above bug is fixed

        # expected = other_scalars // psr
        expected = (other_scalars / gtdi).astype("int64")
        actual = other_scalars // gtdi

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
    ],
)
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
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


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        np.array([10, 20, 30, None, 100]),
        [1000000, 200000, None],
        [1],
        [
            136457654736252,
            134736784364431,
            245345345545332,
            223432411,
            None,
            3634548734,
            None,
        ],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize("dtype", dtypeutils.TIMEDELTA_TYPES)
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

    expected = psr.fillna(fill_value)
    actual = sr.fillna(fill_value)

    assert_eq(expected, actual)
