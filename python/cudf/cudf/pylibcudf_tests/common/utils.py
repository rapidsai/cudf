# Copyright (c) 2024, NVIDIA CORPORATION.

from typing import Optional, Union

import pyarrow as pa
import pytest

from cudf._lib import pylibcudf as plc


def metadata_from_arrow_array(
    pa_array: pa.Array,
) -> Optional[plc.interop.ColumnMetadata]:
    metadata = None
    if pa.types.is_list(dtype := pa_array.type) or pa.types.is_struct(dtype):
        metadata = plc.interop.ColumnMetadata(
            "",
            # libcudf does not store field names, so just match pyarrow's.
            [
                plc.interop.ColumnMetadata(pa_array.type.field(i).name)
                for i in range(pa_array.type.num_fields)
            ],
        )
    return metadata


def assert_column_eq(
    lhs: Union[pa.Array, plc.Column], rhs: Union[pa.Array, plc.Column]
) -> None:
    """Verify that a pylibcudf array and PyArrow array are equal."""
    # Nested types require children metadata to be passed to the conversion function.
    if isinstance(lhs, (pa.Array, pa.ChunkedArray)) and isinstance(
        rhs, plc.Column
    ):
        rhs = plc.interop.to_arrow(
            rhs, metadata=metadata_from_arrow_array(lhs)
        )
    elif isinstance(lhs, plc.Column) and isinstance(
        rhs, (pa.Array, pa.ChunkedArray)
    ):
        lhs = plc.interop.to_arrow(
            lhs, metadata=metadata_from_arrow_array(rhs)
        )
    else:
        raise ValueError(
            "One of the inputs must be a Column and the other an Array"
        )

    if isinstance(lhs, pa.ChunkedArray):
        lhs = lhs.combine_chunks()
    if isinstance(rhs, pa.ChunkedArray):
        rhs = rhs.combine_chunks()

    assert lhs.equals(rhs)


def assert_table_eq(plc_table: plc.Table, pa_table: pa.Table) -> None:
    """Verify that a pylibcudf table and PyArrow table are equal."""
    plc_shape = (plc_table.num_rows(), plc_table.num_columns())
    assert plc_shape == pa_table.shape

    for plc_col, pa_col in zip(plc_table.columns(), pa_table.columns):
        assert_column_eq(pa_col, plc_col)


def cudf_raises(expected_exception: BaseException, *args, **kwargs):
    # A simple wrapper around pytest.raises that defaults to looking for cudf exceptions
    match = kwargs.get("match", None)
    if match is None:
        kwargs["match"] = "CUDF failure at"
    return pytest.raises(expected_exception, *args, **kwargs)


# TODO: Consider moving these type utilities into pylibcudf.types itself.
def is_signed_integer(plc_dtype: plc.DataType):
    return (
        plc.TypeId.INT8.value <= plc_dtype.id().value <= plc.TypeId.INT64.value
    )


def is_unsigned_integer(plc_dtype: plc.DataType):
    return plc_dtype.id() in (
        plc.TypeId.UINT8,
        plc.TypeId.UINT16,
        plc.TypeId.UINT32,
        plc.TypeId.UINT64,
    )


def is_integer(plc_dtype: plc.DataType):
    return plc_dtype.id() in (
        plc.TypeId.INT8,
        plc.TypeId.INT16,
        plc.TypeId.INT32,
        plc.TypeId.INT64,
    )


def is_floating(plc_dtype: plc.DataType):
    return plc_dtype.id() in (
        plc.TypeId.FLOAT32,
        plc.TypeId.FLOAT64,
    )


def is_boolean(plc_dtype: plc.DataType):
    return plc_dtype.id() == plc.TypeId.BOOL8


def is_string(plc_dtype: plc.DataType):
    return plc_dtype.id() == plc.TypeId.STRING


def is_fixed_width(plc_dtype: plc.DataType):
    return (
        is_integer(plc_dtype)
        or is_floating(plc_dtype)
        or is_boolean(plc_dtype)
    )


# We must explicitly specify this type via a field to ensure we don't include
# nullability accidentally.
DEFAULT_STRUCT_TESTING_TYPE = pa.struct(
    [pa.field("v", pa.int64(), nullable=False)]
)
