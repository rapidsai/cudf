# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf._lib.transform import mask_to_bools
from cudf.core.column.column import as_column
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal
from cudf.utils import dtypes as dtypeutils

dtypes = sorted(
    list(
        dtypeutils.ALL_TYPES
        - {
            "datetime64[s]",
            "datetime64[ms]",
            "datetime64[us]",
            "timedelta64[s]",
            "timedelta64[ms]",
            "timedelta64[us]",
        }
    )
)


@pytest.fixture(params=dtypes, ids=dtypes)
def pandas_input(request):
    dtype = request.param
    rng = np.random.default_rng(seed=0)
    size = 100

    def random_ints(dtype, size):
        dtype_min = np.iinfo(dtype).min
        dtype_max = np.iinfo(dtype).max
        rng = np.random.default_rng(seed=0)
        return rng.integers(dtype_min, dtype_max, size=size, dtype=dtype)

    try:
        dtype = np.dtype(dtype)
    except TypeError:
        if dtype == "category":
            data = random_ints(np.int64, size)
        else:
            raise
    else:
        if dtype.kind == "b":
            data = rng.choice([False, True], size=size)
        elif dtype.kind in ("m", "M"):
            # datetime or timedelta
            data = random_ints(np.int64, size)
        elif dtype.kind == "U":
            # Unicode strings of integers like "12345"
            data = random_ints(np.int64, size).astype(dtype.str)
        elif dtype.kind == "f":
            # floats in [0.0, 1.0)
            data = rng.random(size=size, dtype=dtype)
        else:
            data = random_ints(dtype, size)
    return pd.Series(data, dtype=dtype)


def str_host_view(list_of_str, to_dtype):
    return np.concatenate(
        [np.frombuffer(s.encode("utf-8"), dtype=to_dtype) for s in list_of_str]
    )


@pytest.mark.parametrize("offset", [0, 1, 15])
@pytest.mark.parametrize("size", [50, 10, 0])
def test_column_offset_and_size(pandas_input, offset, size):
    col = cudf.core.column.as_column(pandas_input)
    col = cudf.core.column.build_column(
        data=col.base_data,
        dtype=col.dtype,
        mask=col.base_mask,
        size=size,
        offset=offset,
        children=col.base_children,
    )

    if isinstance(col.dtype, cudf.CategoricalDtype):
        assert col.size == col.codes.size
        assert col.size == (col.codes.data.size / col.codes.dtype.itemsize)
    elif cudf.api.types.is_string_dtype(col.dtype):
        if col.size > 0:
            assert col.size == (col.children[0].size - 1)
            assert col.size == (
                (col.children[0].data.size / col.children[0].dtype.itemsize)
                - 1
            )
    else:
        assert col.size == (col.data.size / col.dtype.itemsize)

    got = cudf.Series._from_column(col)

    if offset is None:
        offset = 0
    if size is None:
        size = 100
    else:
        size = size + offset

    slicer = slice(offset, size)
    expect = pandas_input.iloc[slicer].reset_index(drop=True)

    assert_eq(expect, got)


def column_slicing_test(col, offset, size, cast_to_float=False):
    col_slice = col.slice(offset, offset + size)
    series = cudf.Series._from_column(col)
    sliced_series = cudf.Series._from_column(col_slice)

    if cast_to_float:
        pd_series = series.astype(float).to_pandas()
        sliced_series = sliced_series.astype(float)
    else:
        pd_series = series.to_pandas()

    if isinstance(col.dtype, cudf.CategoricalDtype):
        # The cudf.Series is constructed from an already sliced column, whereas
        # the pandas.Series is constructed from the unsliced series and then
        # sliced, so the indexes should be different and we must ignore it.
        # However, we must compare these as frames, not raw arrays,  because
        # numpy comparison of categorical values won't work.
        assert_eq(
            pd_series[offset : offset + size].reset_index(drop=True),
            sliced_series.reset_index(drop=True),
        )
    else:
        assert_eq(
            np.asarray(pd_series[offset : offset + size]),
            sliced_series.to_numpy(),
        )


@pytest.mark.parametrize("offset", [0, 1, 15])
@pytest.mark.parametrize("size", [50, 10, 0])
def test_column_slicing(pandas_input, offset, size):
    col = cudf.core.column.as_column(pandas_input)
    column_slicing_test(col, offset, size)


@pytest.mark.parametrize("offset", [0, 1, 15])
@pytest.mark.parametrize("size", [50, 10, 0])
@pytest.mark.parametrize("precision", [2, 3, 5])
@pytest.mark.parametrize("scale", [0, 1, 2])
@pytest.mark.parametrize(
    "decimal_type",
    [cudf.Decimal128Dtype, cudf.Decimal64Dtype, cudf.Decimal32Dtype],
)
def test_decimal_column_slicing(offset, size, precision, scale, decimal_type):
    col = cudf.core.column.as_column(
        pd.Series(np.random.default_rng(seed=0).random(1000))
    )
    col = col.astype(decimal_type(precision, scale))
    column_slicing_test(col, offset, size, True)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[23, 68, 2, 38, 9, 83, 72, 6, 98, 30]]),
        np.array([[1, 2], [7, 6]]),
    ],
)
def test_column_series_multi_dim(data):
    with pytest.raises(ValueError):
        cudf.Series(data)

    with pytest.raises(ValueError):
        cudf.core.column.as_column(data)


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ([1, "1.0", "2", -3], cudf.errors.MixedTypeError),
        ([np.nan, 0, "null", cp.nan], cudf.errors.MixedTypeError),
        (
            [np.int32(4), np.float64(1.5), np.float32(1.290994), np.int8(0)],
            None,
        ),
    ],
)
def test_column_mixed_dtype(data, error):
    if error is None:
        cudf.Series(data)
    else:
        with pytest.raises(error):
            cudf.Series(data)


@pytest.mark.parametrize("nan_as_null", [True, False])
@pytest.mark.parametrize(
    "scalar",
    [np.nan, pd.Timedelta(days=1), pd.Timestamp(2020, 1, 1)],
    ids=repr,
)
@pytest.mark.parametrize("size", [1, 10])
def test_as_column_scalar_with_nan(nan_as_null, scalar, size):
    expected = (
        cudf.Series([scalar] * size, nan_as_null=nan_as_null)
        .dropna()
        .to_numpy()
    )

    got = (
        cudf.Series._from_column(
            as_column(scalar, length=size, nan_as_null=nan_as_null)
        )
        .dropna()
        .to_numpy()
    )

    np.testing.assert_equal(expected, got)


@pytest.mark.parametrize("data", [[1.1, 2.2, 3.3, 4.4], [1, 2, 3, 4]])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_column_series_cuda_array_dtype(data, dtype):
    psr = pd.Series(np.asarray(data), dtype=dtype)
    sr = cudf.Series(cp.asarray(data), dtype=dtype)

    assert_eq(psr, sr)

    psr = pd.Series(data, dtype=dtype)
    sr = cudf.Series(data, dtype=dtype)

    assert_eq(psr, sr)


def test_column_zero_length_slice():
    # see https://github.com/rapidsai/cudf/pull/4777
    from numba import cuda

    x = cudf.DataFrame({"a": [1]})
    the_column = x[1:]["a"]._column

    expect = np.array([], dtype="int8")
    got = cuda.as_cuda_array(the_column.data).copy_to_host()

    np.testing.assert_array_equal(expect, got)


def test_column_chunked_array_creation():
    pyarrow_array = pa.array([1, 2, 3] * 1000)
    chunked_array = pa.chunked_array(pyarrow_array)

    actual_column = cudf.core.column.as_column(chunked_array, dtype="float")
    expected_column = cudf.core.column.as_column(pyarrow_array, dtype="float")

    assert_eq(
        cudf.Series._from_column(actual_column),
        cudf.Series._from_column(expected_column),
    )

    actual_column = cudf.core.column.as_column(chunked_array)
    expected_column = cudf.core.column.as_column(pyarrow_array)

    assert_eq(
        cudf.Series._from_column(actual_column),
        cudf.Series._from_column(expected_column),
    )


@pytest.mark.parametrize(
    "data,from_dtype,to_dtype",
    [
        # equal size different kind
        (np.arange(3), "int64", "float64"),
        (np.arange(3), "float32", "int32"),
        (np.arange(1), "int64", "datetime64[ns]"),
        # size / 2^n should work for all n
        (np.arange(3), "int64", "int32"),
        (np.arange(3), "int64", "int16"),
        (np.arange(3), "int64", "int8"),
        (np.arange(3), "float64", "float32"),
        # evenly divides into bigger type
        (np.arange(8), "int8", "int64"),
        (np.arange(16), "int8", "int64"),
        (np.arange(128), "int8", "int64"),
        (np.arange(2), "float32", "int64"),
        (np.arange(8), "int8", "datetime64[ns]"),
        (np.arange(16), "int8", "datetime64[ns]"),
    ],
)
def test_column_view_valid_numeric_to_numeric(data, from_dtype, to_dtype):
    cpu_data = np.asarray(data, dtype=from_dtype)
    gpu_data = as_column(data, dtype=from_dtype)

    cpu_data_view = cpu_data.view(to_dtype)
    gpu_data_view = gpu_data.view(to_dtype)

    expect = pd.Series(cpu_data_view, dtype=cpu_data_view.dtype)
    got = cudf.Series._from_column(gpu_data_view).astype(gpu_data_view.dtype)

    gpu_ptr = gpu_data.data.get_ptr(mode="read")
    assert gpu_ptr == got._column.data.get_ptr(mode="read")
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data,from_dtype,to_dtype",
    [
        (np.arange(9), "int8", "int64"),
        (np.arange(3), "int8", "int16"),
        (np.arange(6), "int8", "float32"),
        (np.arange(1), "int8", "datetime64[ns]"),
    ],
)
def test_column_view_invalid_numeric_to_numeric(data, from_dtype, to_dtype):
    cpu_data = np.asarray(data, dtype=from_dtype)
    gpu_data = as_column(data, dtype=from_dtype)

    assert_exceptions_equal(
        lfunc=cpu_data.view,
        rfunc=gpu_data.view,
        lfunc_args_and_kwargs=([to_dtype],),
        rfunc_args_and_kwargs=([to_dtype],),
    )


@pytest.mark.parametrize(
    "data,to_dtype",
    [
        (["a", "b", "c"], "int8"),
        (["ab"], "int8"),
        (["ab"], "int16"),
        (["a", "ab", "a"], "int8"),
        (["abcd", "efgh"], "float32"),
        (["abcdefgh"], "datetime64[ns]"),
    ],
)
def test_column_view_valid_string_to_numeric(data, to_dtype):
    expect = cudf.Series._from_column(cudf.Series(data)._column.view(to_dtype))
    got = cudf.Series(str_host_view(data, to_dtype))

    assert_eq(expect, got)


def test_column_view_nulls_widths_even():
    data = [1, 2, None, 4, None]
    expect_data = [
        np.int32(val).view("float32") if val is not None else np.nan
        for val in data
    ]

    sr = cudf.Series(data, dtype="int32")
    expect = cudf.Series(expect_data, dtype="float32")
    got = cudf.Series._from_column(sr._column.view("float32"))

    assert_eq(expect, got)

    data = [None, 2.1, None, 5.3, 8.8]
    expect_data = [
        np.float64(val).view("int64") if val is not None else val
        for val in data
    ]

    sr = cudf.Series(data, dtype="float64")
    expect = cudf.Series(expect_data, dtype="int64")
    got = cudf.Series._from_column(sr._column.view("int64"))

    assert_eq(expect, got)


@pytest.mark.parametrize("slc", [slice(1, 5), slice(0, 4), slice(2, 4)])
def test_column_view_numeric_slice(slc):
    data = np.array([1, 2, 3, 4, 5], dtype="int32")
    sr = cudf.Series(data)

    expect = cudf.Series(data[slc].view("int64"))
    got = cudf.Series._from_column(
        sr._column.slice(slc.start, slc.stop).view("int64")
    )

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "slc", [slice(3, 5), slice(0, 4), slice(2, 5), slice(1, 3)]
)
def test_column_view_string_slice(slc):
    data = ["a", "bcde", "cd", "efg", "h"]

    expect = cudf.Series._from_column(
        cudf.Series(data)._column.slice(slc.start, slc.stop).view("int8")
    )
    got = cudf.Series(str_host_view(data[slc], "int8"))

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            np.array([1, 2, 3, 4, 5], dtype="uint8"),
            cudf.core.column.as_column([1, 2, 3, 4, 5], dtype="uint8"),
        ),
        (
            cp.array([1, 2, 3, 4, 5], dtype="uint8"),
            cudf.core.column.as_column([1, 2, 3, 4, 5], dtype="uint8"),
        ),
        (
            cp.array([], dtype="uint8"),
            cudf.core.column.column_empty(0, dtype="uint8"),
        ),
        (
            cp.array([255], dtype="uint8"),
            cudf.core.column.as_column([255], dtype="uint8"),
        ),
    ],
)
def test_as_column_buffer(data, expected):
    actual_column = cudf.core.column.as_column(
        cudf.core.buffer.as_buffer(data), dtype=data.dtype
    )
    assert_eq(
        cudf.Series._from_column(actual_column),
        cudf.Series._from_column(expected),
    )


@pytest.mark.parametrize(
    "data,pyarrow_kwargs,cudf_kwargs",
    [
        (
            [100, 200, 300],
            {"type": pa.decimal128(3)},
            {"dtype": cudf.core.dtypes.Decimal128Dtype(3, 0)},
        ),
        (
            [{"a": 1, "b": 3}, {"c": 2, "d": 4}],
            {},
            {},
        ),
        (
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            {},
            {},
        ),
    ],
)
def test_as_column_arrow_array(data, pyarrow_kwargs, cudf_kwargs):
    pyarrow_data = pa.array(data, **pyarrow_kwargs)
    cudf_from_pyarrow = as_column(pyarrow_data)
    expected = as_column(data, **cudf_kwargs)
    assert_eq(
        cudf.Series._from_column(cudf_from_pyarrow),
        cudf.Series._from_column(expected),
    )


@pytest.mark.parametrize(
    "pd_dtype,expect_dtype",
    [
        # TODO: Nullable float is coming
        (pd.StringDtype(), np.dtype("O")),
        (pd.UInt8Dtype(), np.dtype("uint8")),
        (pd.UInt16Dtype(), np.dtype("uint16")),
        (pd.UInt32Dtype(), np.dtype("uint32")),
        (pd.UInt64Dtype(), np.dtype("uint64")),
        (pd.Int8Dtype(), np.dtype("int8")),
        (pd.Int16Dtype(), np.dtype("int16")),
        (pd.Int32Dtype(), np.dtype("int32")),
        (pd.Int64Dtype(), np.dtype("int64")),
        (pd.BooleanDtype(), np.dtype("bool")),
    ],
)
def test_build_df_from_nullable_pandas_dtype(pd_dtype, expect_dtype):
    if pd_dtype == pd.StringDtype():
        data = ["a", pd.NA, "c", pd.NA, "e"]
    elif pd_dtype == pd.BooleanDtype():
        data = [True, pd.NA, False, pd.NA, True]
    else:
        data = [1, pd.NA, 3, pd.NA, 5]

    pd_data = pd.DataFrame.from_dict({"a": data}, dtype=pd_dtype)
    gd_data = cudf.DataFrame.from_pandas(pd_data)

    assert gd_data["a"].dtype == expect_dtype

    # check mask
    expect_mask = [x is not pd.NA for x in pd_data["a"]]
    got_mask = mask_to_bools(
        gd_data["a"]._column.base_mask, 0, len(gd_data)
    ).values_host

    np.testing.assert_array_equal(expect_mask, got_mask)


@pytest.mark.parametrize(
    "pd_dtype,expect_dtype",
    [
        # TODO: Nullable float is coming
        (pd.StringDtype(), np.dtype("O")),
        (pd.UInt8Dtype(), np.dtype("uint8")),
        (pd.UInt16Dtype(), np.dtype("uint16")),
        (pd.UInt32Dtype(), np.dtype("uint32")),
        (pd.UInt64Dtype(), np.dtype("uint64")),
        (pd.Int8Dtype(), np.dtype("int8")),
        (pd.Int16Dtype(), np.dtype("int16")),
        (pd.Int32Dtype(), np.dtype("int32")),
        (pd.Int64Dtype(), np.dtype("int64")),
        (pd.BooleanDtype(), np.dtype("bool")),
    ],
)
def test_build_series_from_nullable_pandas_dtype(pd_dtype, expect_dtype):
    if pd_dtype == pd.StringDtype():
        data = ["a", pd.NA, "c", pd.NA, "e"]
    elif pd_dtype == pd.BooleanDtype():
        data = [True, pd.NA, False, pd.NA, True]
    else:
        data = [1, pd.NA, 3, pd.NA, 5]

    pd_data = pd.Series(data, dtype=pd_dtype)
    gd_data = cudf.Series.from_pandas(pd_data)

    assert gd_data.dtype == expect_dtype

    # check mask
    expect_mask = [x is not pd.NA for x in pd_data]
    got_mask = mask_to_bools(
        gd_data._column.base_mask, 0, len(gd_data)
    ).values_host

    np.testing.assert_array_equal(expect_mask, got_mask)


@pytest.mark.parametrize(
    "alias,expect_dtype",
    [
        ("UInt8", "uint8"),
        ("UInt16", "uint16"),
        ("UInt32", "uint32"),
        ("UInt64", "uint64"),
        ("Int8", "int8"),
        ("Int16", "int16"),
        ("Int32", "int32"),
        ("Int64", "int64"),
        ("boolean", "bool"),
        ("Float32", "float32"),
        ("Float64", "float64"),
    ],
)
@pytest.mark.parametrize(
    "data",
    [[1, 2, 0]],
)
def test_astype_with_aliases(alias, expect_dtype, data):
    pd_data = pd.Series(data)
    gd_data = cudf.Series.from_pandas(pd_data)

    assert_eq(pd_data.astype(expect_dtype), gd_data.astype(alias))
