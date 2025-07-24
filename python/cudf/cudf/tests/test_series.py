# Copyright (c) 2025, NVIDIA CORPORATION.
import decimal
import operator

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.column.column import as_column
from cudf.errors import MixedTypeError
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
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


def test_fill_new_category():
    gs = cudf.Series(pd.Categorical(["a", "b", "c"]))
    with pytest.raises(TypeError):
        gs[0:1] = "d"


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


@pytest.mark.parametrize("data", [[True, False, None], [10, 200, 300]])
@pytest.mark.parametrize("index", [None, [10, 20, 30]])
def test_series_contains(data, index):
    ps = pd.Series(data, index=index)
    gs = cudf.Series(data, index=index)

    assert_eq(1 in ps, 1 in gs)
    assert_eq(10 in ps, 10 in gs)
    assert_eq(True in ps, True in gs)
    assert_eq(False in ps, False in gs)


@pytest.mark.parametrize("attr", ["nlargest", "nsmallest"])
def test_series_nlargest_nsmallest_str_error(attr):
    gs = cudf.Series(["a", "b", "c", "d", "e"])
    ps = gs.to_pandas()

    assert_exceptions_equal(
        getattr(gs, attr), getattr(ps, attr), ([], {"n": 1}), ([], {"n": 1})
    )


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
def test_from_pandas_object_dtype_passed_dtype(klass):
    result = klass(pd.Series([True, False], dtype=object), dtype="int8")
    expected = klass(pa.array([1, 0], type=pa.int8()))
    assert_eq(result, expected)


def test_series_setitem_mixed_bool_dtype():
    s = cudf.Series([True, False, True])
    with pytest.raises(TypeError):
        s[0] = 10


def test_series_duplicate_index_reindex():
    gs = cudf.Series([0, 1, 2, 3], index=[0, 0, 1, 1])
    ps = gs.to_pandas()

    assert_exceptions_equal(
        gs.reindex,
        ps.reindex,
        lfunc_args_and_kwargs=([10, 11, 12, 13], {}),
        rfunc_args_and_kwargs=([10, 11, 12, 13], {}),
    )


@pytest.mark.parametrize("value", [1, 1.1])
def test_nans_to_nulls_noop_copies_column(value):
    ser1 = cudf.Series([value])
    ser2 = ser1.nans_to_nulls()
    assert ser1._column is not ser2._column


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
