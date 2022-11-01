# Copyright (c) 2021-2022, NVIDIA CORPORATION.

from typing import Any, Tuple

import cupy as cp
import pandas as pd
import pytest

import cudf
from cudf.core.buffer import Buffer
from cudf.core.column import build_column
from cudf.core.df_protocol import (
    DataFrameObject,
    _CuDFBuffer,
    _CuDFColumn,
    _DtypeKind,
    _from_dataframe,
    protocol_dtype_to_cupy_dtype,
)
from cudf.testing._utils import assert_eq


def assert_buffer_equal(buffer_and_dtype: Tuple[_CuDFBuffer, Any], cudfcol):
    buf, dtype = buffer_and_dtype
    device_id = cp.asarray(cudfcol.data).device.id
    assert buf.__dlpack_device__() == (2, device_id)
    col_from_buf = build_column(
        Buffer(data=buf.ptr, size=buf.bufsize, owner=None),
        protocol_dtype_to_cupy_dtype(dtype),
    )
    # check that non null values are the equals as nulls are represented
    # by sentinel values in the buffer.
    # FIXME: In gh-10202 some minimal fixes were added to unblock CI. But
    # currently only non-null values are compared, null positions are
    # unchecked.
    non_null_idxs = ~cudf.Series(cudfcol).isna()
    assert_eq(
        col_from_buf.apply_boolean_mask(non_null_idxs),
        cudfcol.apply_boolean_mask(non_null_idxs),
    )

    if dtype[0] != _DtypeKind.BOOL:
        array_from_dlpack = cp.fromDlpack(buf.__dlpack__()).get()
        col_array = cp.asarray(cudfcol.data_array_view).get()
        assert_eq(
            array_from_dlpack[non_null_idxs.to_numpy()].flatten(),
            col_array[non_null_idxs.to_numpy()].flatten(),
        )
    else:
        pytest.raises(TypeError, buf.__dlpack__)


def assert_column_equal(col: _CuDFColumn, cudfcol):
    assert col.size == cudfcol.size
    assert col.offset == 0
    assert col.null_count == cudfcol.null_count
    assert col.num_chunks() == 1
    if col.null_count == 0:
        pytest.raises(RuntimeError, col._get_validity_buffer)
        assert col.get_buffers()["validity"] is None
    else:
        assert_buffer_equal(
            col.get_buffers()["validity"],
            cudfcol._get_mask_as_column().astype(cp.uint8),
        )

    if col.dtype[0] == _DtypeKind.CATEGORICAL:
        assert_buffer_equal(col.get_buffers()["data"], cudfcol.codes)
        assert col.get_buffers()["offsets"] is None

    elif col.dtype[0] == _DtypeKind.STRING:
        assert_buffer_equal(col.get_buffers()["data"], cudfcol.children[1])
        assert_buffer_equal(col.get_buffers()["offsets"], cudfcol.children[0])

    else:
        assert_buffer_equal(col.get_buffers()["data"], cudfcol)
        assert col.get_buffers()["offsets"] is None

    if col.null_count == 0:
        assert col.describe_null == (0, None)
    else:
        assert col.describe_null == (3, 0)


def assert_dataframe_equal(dfo: DataFrameObject, df: cudf.DataFrame):
    assert dfo.num_columns() == len(df.columns)
    assert dfo.num_rows() == len(df)
    assert dfo.num_chunks() == 1
    assert dfo.column_names() == tuple(df.columns)
    for col in df.columns:
        assert_column_equal(dfo.get_column_by_name(col), df[col]._column)


def assert_from_dataframe_equals(dfobj):
    df2 = _from_dataframe(dfobj)

    assert_dataframe_equal(dfobj, df2)
    if isinstance(dfobj._df, cudf.DataFrame):
        assert_eq(dfobj._df, df2)

    elif isinstance(dfobj._df, pd.DataFrame):
        assert_eq(cudf.DataFrame(dfobj._df), df2)

    else:
        raise TypeError(f"{type(dfobj._df)} not supported yet.")


def assert_from_dataframe_exception(dfobj):
    exception_msg = "This operation must copy data from CPU to GPU."
    " Set `allow_copy=True` to allow it."
    with pytest.raises(TypeError, match=exception_msg):
        _from_dataframe(dfobj)


def assert_df_unique_dtype_cols(data):
    cdf = cudf.DataFrame(data=data)
    assert_from_dataframe_equals(cdf.__dataframe__(allow_copy=False))
    assert_from_dataframe_equals(cdf.__dataframe__(allow_copy=True))


def test_from_dataframe():
    data = dict(a=[1, 2, 3], b=[9, 10, 11])
    df1 = cudf.DataFrame(data=data)
    df2 = cudf.from_dataframe(df1)
    assert_eq(df1, df2)

    df3 = cudf.from_dataframe(df2)
    assert_eq(df1, df3)


def test_int_dtype():
    data_int = dict(a=[1, 2, 3], b=[9, 10, 11])
    assert_df_unique_dtype_cols(data_int)


def test_float_dtype():
    data_float = dict(a=[1.5, 2.5, 3.5], b=[9.2, 10.5, 11.8])
    assert_df_unique_dtype_cols(data_float)


def test_categorical_dtype():
    cdf = cudf.DataFrame({"A": [1, 2, 5, 1]})
    cdf["A"] = cdf["A"].astype("category")
    col = cdf.__dataframe__().get_column_by_name("A")
    assert col.dtype[0] == _DtypeKind.CATEGORICAL
    assert col.describe_categorical == (False, True, {0: 1, 1: 2, 2: 5})
    assert_from_dataframe_equals(cdf.__dataframe__(allow_copy=False))
    assert_from_dataframe_equals(cdf.__dataframe__(allow_copy=True))


def test_bool_dtype():
    data_bool = dict(a=[True, True, False], b=[False, True, False])
    assert_df_unique_dtype_cols(data_bool)


def test_string_dtype():
    data_string = dict(a=["a", "b", "cdef", "", "g"])
    assert_df_unique_dtype_cols(data_string)


def test_mixed_dtype():
    data_mixed = dict(
        int=[1, 2, 3],
        float=[1.5, 2.5, 3.5],
        bool=[True, False, True],
        categorical=[5, 1, 5],
        string=["rapidsai-cudf ", "", "df protocol"],
    )
    assert_df_unique_dtype_cols(data_mixed)


def test_NA_int_dtype():
    data_int = dict(
        a=[1, None, 3, None, 5],
        b=[9, 10, None, 7, 8],
        c=[6, 19, 20, 100, 1000],
    )
    assert_df_unique_dtype_cols(data_int)


def test_NA_float_dtype():
    data_float = dict(
        a=[1.4, None, 3.6, None, 5.2],
        b=[9.7, 10.9, None, 7.8, 8.2],
        c=[6.1, 19.2, 20.3, 100.4, 1000.5],
    )
    assert_df_unique_dtype_cols(data_float)


def test_NA_categorical_dtype():
    df = cudf.DataFrame({"A": [1, 2, 5, 1]})
    df["B"] = df["A"].astype("category")
    df.at[[1, 3], "B"] = None  # Set two items to null

    # Some detailed testing for correctness of dtype and null handling:
    col = df.__dataframe__().get_column_by_name("B")
    assert col.dtype[0] == _DtypeKind.CATEGORICAL
    assert col.null_count == 2
    assert col.describe_null == (3, 0)
    assert col.num_chunks() == 1
    assert col.describe_categorical == (False, True, {0: 1, 1: 2, 2: 5})
    assert_from_dataframe_equals(df.__dataframe__(allow_copy=False))
    assert_from_dataframe_equals(df.__dataframe__(allow_copy=True))


def test_NA_bool_dtype():
    data_bool = dict(a=[None, True, False], b=[False, None, None])
    assert_df_unique_dtype_cols(data_bool)


def test_NA_string_dtype():
    df = cudf.DataFrame({"A": ["a", "b", "cdef", "", "g"]})
    df["B"] = df["A"].astype("object")
    df.at[1, "B"] = cudf.NA  # Set one item to null

    # Test for correctness and null handling:
    col = df.__dataframe__().get_column_by_name("B")
    assert col.dtype[0] == _DtypeKind.STRING
    assert col.null_count == 1
    assert col.describe_null == (3, 0)
    assert col.num_chunks() == 1
    assert_from_dataframe_equals(df.__dataframe__(allow_copy=False))
    assert_from_dataframe_equals(df.__dataframe__(allow_copy=True))


def test_NA_mixed_dtype():
    data_mixed = dict(
        int=[1, None, 2, 3, 1000],
        float=[None, 1.5, 2.5, 3.5, None],
        bool=[True, None, False, None, None],
        categorical=[5, 1, 5, 3, None],
        string=[None, None, None, "df protocol", None],
    )
    assert_df_unique_dtype_cols(data_mixed)
