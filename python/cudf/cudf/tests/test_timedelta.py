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
