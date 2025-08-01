# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import decimal

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.column.column import as_column
from cudf.errors import MixedTypeError
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


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
)
def timedelta_data(request):
    return request.param


def test_timedelta_series_create(timedelta_data, timedelta_types_as_str):
    if timedelta_types_as_str != "timedelta64[ns]":
        pytest.skip(
            "Bug in pandas : https://github.com/pandas-dev/pandas/issues/35465"
        )
    psr = pd.Series(
        cp.asnumpy(timedelta_data)
        if isinstance(timedelta_data, cp.ndarray)
        else timedelta_data,
        dtype=timedelta_types_as_str,
    )
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)

    assert_eq(psr, gsr)


def test_timedelta_from_pandas(timedelta_data, timedelta_types_as_str):
    psr = pd.Series(
        cp.asnumpy(timedelta_data)
        if isinstance(timedelta_data, cp.ndarray)
        else timedelta_data,
        dtype=timedelta_types_as_str,
    )
    gsr = cudf.from_pandas(psr)

    assert_eq(psr, gsr)


def test_construct_int_series_with_nulls_compat_mode():
    # in compatibility mode, constructing a Series
    # with nulls should result in a floating Series:
    with cudf.option_context("mode.pandas_compatible", True):
        s = cudf.Series([1, 2, None])
    assert s.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 24, "d": 1010},
        {"a": 1},
        {1: "a", 2: "b", 24: "c", 1010: "d"},
        {1: "a"},
        {"a": [1]},
    ],
)
def test_series_init_dict(data):
    pandas_series = pd.Series(data)
    cudf_series = cudf.Series(data)

    assert_eq(pandas_series, cudf_series)


def test_series_unitness_np_datetimelike_units():
    data = np.array([np.timedelta64(1)])
    with pytest.raises(TypeError):
        cudf.Series(data)
    with pytest.raises(TypeError):
        pd.Series(data)


def test_list_category_like_maintains_dtype():
    dtype = cudf.CategoricalDtype(categories=[1, 2, 3, 4], ordered=True)
    data = [1, 2, 3]
    result = cudf.Series._from_column(as_column(data, dtype=dtype))
    expected = pd.Series(data, dtype=dtype.to_pandas())
    assert_eq(result, expected)


def test_list_interval_like_maintains_dtype():
    dtype = cudf.IntervalDtype(subtype=np.int8)
    data = [pd.Interval(1, 2)]
    result = cudf.Series._from_column(as_column(data, dtype=dtype))
    expected = pd.Series(data, dtype=dtype.to_pandas())
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "klass", [cudf.Series, cudf.Index, pd.Series, pd.Index]
)
def test_series_from_named_object_name_priority(klass):
    result = cudf.Series(klass([1], name="a"), name="b")
    assert result.name == "b"


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 3},
        cudf.Series([1, 2, 3], index=list("abc")),
        pd.Series([1, 2, 3], index=list("abc")),
    ],
)
def test_series_from_object_with_index_index_arg_reindex(data):
    result = cudf.Series(data, index=list("bca"))
    expected = cudf.Series([2, 3, 1], index=list("bca"))
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        {0: 1, 1: 2, 2: 3},
        cudf.Series([1, 2, 3]),
        cudf.Index([1, 2, 3]),
        pd.Series([1, 2, 3]),
        pd.Index([1, 2, 3]),
        [1, 2, 3],
    ],
)
def test_series_dtype_astypes(data):
    result = cudf.Series(data, dtype="float64")
    expected = cudf.Series([1.0, 2.0, 3.0])
    assert_eq(result, expected)


@pytest.mark.parametrize("pa_type", [pa.string, pa.large_string])
def test_series_from_large_string(pa_type):
    pa_string_array = pa.array(["a", "b", "c"]).cast(pa_type())
    got = cudf.Series(pa_string_array)
    expected = pd.Series(pa_string_array)

    assert_eq(expected, got)


def test_series_init_with_nans():
    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.Series([1, 2, 3, np.nan])
    assert gs.dtype == np.dtype("float64")
    ps = pd.Series([1, 2, 3, np.nan])
    assert_eq(ps, gs)


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2, 3], [10, 20]],
        [[1.0, 2.0, 3.0], None, [10.0, 20.0, np.nan]],
        [[5, 6], None, [1]],
        [None, None, None, None, None, [10, 20]],
    ],
)
@pytest.mark.parametrize("klass", [cudf.Series, list, cp.array])
def test_nested_series_from_sequence_data(data, klass):
    actual = cudf.Series(
        [klass(val) if val is not None else val for val in data]
    )
    expected = cudf.Series(data)
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        lambda: cp.ones(5, dtype=cp.float16),
        lambda: np.ones(5, dtype="float16"),
        lambda: pd.Series([0.1, 1.2, 3.3], dtype="float16"),
        pytest.param(
            lambda: pa.array(np.ones(5, dtype="float16")),
            marks=pytest.mark.xfail(
                reason="https://issues.apache.org/jira/browse/ARROW-13762"
            ),
        ),
    ],
)
def test_series_raises_float16(data):
    data = data()
    with pytest.raises(TypeError):
        cudf.Series(data)


@pytest.mark.parametrize(
    "data", [[True, False, None, True, False], [None, None], []]
)
@pytest.mark.parametrize("bool_dtype", ["bool", "boolean", pd.BooleanDtype()])
def test_nullable_bool_dtype_series(data, bool_dtype):
    psr = pd.Series(data, dtype=pd.BooleanDtype())
    gsr = cudf.Series(data, dtype=bool_dtype)

    assert_eq(psr, gsr.to_pandas(nullable=True))


@pytest.mark.parametrize("data", [None, 123, 33243243232423, 0])
@pytest.mark.parametrize("klass", [pd.Timestamp, pd.Timedelta])
def test_temporal_scalar_series_init(data, klass):
    scalar = klass(data)
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


def test_int8_int16_construction():
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


def test_int64_equality():
    s = cudf.Series(np.asarray([2**63 - 10, 2**63 - 100], dtype=np.int64))
    assert (s != np.int64(2**63 - 1)).all()


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


def test_series_data_and_index_length_mismatch():
    assert_exceptions_equal(
        lfunc=pd.Series,
        rfunc=cudf.Series,
        lfunc_args_and_kwargs=([], {"data": [11], "index": [10, 11]}),
        rfunc_args_and_kwargs=([], {"data": [11], "index": [10, 11]}),
    )


@pytest.mark.parametrize(
    "dtype", ["datetime64[ns]", "timedelta64[ns]", "object", "str"]
)
def test_series_mixed_dtype_error(dtype):
    ps = pd.concat([pd.Series([1, 2, 3], dtype=dtype), pd.Series([10, 11])])
    with pytest.raises(TypeError):
        cudf.Series(ps)
    with pytest.raises(TypeError):
        cudf.Series(ps.array)


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


def test_series_init_from_series_and_index():
    ser = cudf.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
    result = cudf.Series(ser, index=list("abcd"))
    expected = cudf.Series([-5, 7, 3, 4], index=list("abcd"))
    assert_eq(result, expected)


def test_series_constructor_unbounded_sequence():
    class A:
        def __getitem__(self, key):
            return 1

    with pytest.raises(TypeError):
        cudf.Series(A())


def test_series_constructor_error_mixed_type():
    with pytest.raises(MixedTypeError):
        cudf.Series(["abc", np.nan, "123"], nan_as_null=False)


def test_series_from_pandas_sparse():
    pser = pd.Series(range(2), dtype=pd.SparseDtype(np.int64, 0))
    with pytest.raises(NotImplementedError):
        cudf.Series(pser)


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


@pytest.mark.parametrize("base_name", [None, "a"])
def test_series_to_frame_none_name(base_name):
    result = cudf.Series(range(1), name=base_name).to_frame(name=None)
    expected = pd.Series(range(1), name=base_name).to_frame(name=None)
    assert_eq(result, expected)


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


def test_null_like_to_nan_pandas_compat():
    with cudf.option_context("mode.pandas_compatible", True):
        ser = cudf.Series([1, 2, np.nan, 10, None])
        pser = pd.Series([1, 2, np.nan, 10, None])

        assert pser.dtype == ser.dtype
        assert_eq(ser, pser)


def test_non_strings_dtype_object_pandas_compat_raises():
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(TypeError):
            cudf.Series([1], dtype=object)


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


def test_series_empty_dtype():
    expected = pd.Series([])
    actual = cudf.Series([])
    assert_eq(expected, actual, check_dtype=True)


@pytest.mark.parametrize("data", [None, {}, []])
def test_series_empty_index_rangeindex(data):
    expected = cudf.RangeIndex(0)
    result = cudf.Series(data).index
    assert_eq(result, expected)


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


@pytest.mark.parametrize("klass", [cudf.Series, cudf.Index])
def test_from_pandas_object_dtype_passed_dtype(klass):
    result = klass(pd.Series([True, False], dtype=object), dtype="int8")
    expected = klass(pa.array([1, 0], type=pa.int8()))
    assert_eq(result, expected)


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


@pytest.mark.parametrize(
    "ps",
    [
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
    ],
)
def test_roundtrip_series_plc_column(ps):
    expect = cudf.Series(ps)
    actual = cudf.Series.from_pylibcudf(*expect.to_pylibcudf())
    assert_eq(expect, actual)


def test_series_construction_with_nulls():
    fields = [
        pa.array([1], type=pa.int64()),
        pa.array([None], type=pa.int64()),
        pa.array([3], type=pa.int64()),
    ]
    expect = pa.StructArray.from_arrays(fields, ["a", "b", "c"])
    got = cudf.Series(expect).to_arrow()

    assert expect == got


@pytest.mark.parametrize(
    "data",
    [
        [{}],
        [{"a": None}],
        [{"a": 1}],
        [{"a": "one"}],
        [{"a": 1}, {"a": 2}],
        [{"a": 1, "b": "one"}, {"a": 2, "b": "two"}],
        [{"b": "two", "a": None}, None, {"a": "one", "b": "two"}],
    ],
)
def test_create_struct_series(data):
    expect = pd.Series(data)
    got = cudf.Series(data)
    assert_eq(expect, got, check_dtype=False)
