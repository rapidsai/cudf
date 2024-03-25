# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest

from cudf._lib import pylibcudf as plc


def metadata_from_arrow_array(pa_array):
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


def assert_array_eq(plc_column, pa_array):
    """Verify that the pylibcudf array and PyArrow array are equal."""
    # Nested types require children metadata to be passed to the conversion function.
    plc_pa = plc.interop.to_arrow(
        plc_column, metadata=metadata_from_arrow_array(pa_array)
    )

    if isinstance(plc_pa, pa.ChunkedArray):
        plc_pa = plc_pa.combine_chunks()
    if isinstance(pa_array, pa.ChunkedArray):
        pa_array = pa_array.combine_chunks()

    assert plc_pa.equals(pa_array)


def assert_table_eq(plc_table, pa_table):
    """Verify that the pylibcudf array and PyArrow array are equal."""
    plc_shape = (plc_table.num_rows(), plc_table.num_columns())
    assert plc_shape == pa_table.shape

    for plc_col, pa_col in zip(plc_table.columns(), pa_table.columns):
        assert_array_eq(plc_col, pa_col)


def cudf_raises(expected_exception, *args, **kwargs):
    # A simple wrapper around pytest.raises that defaults to looking for cudf exceptions
    match = kwargs.get("match", None)
    if match is None:
        kwargs["match"] = "CUDF failure at"
    return pytest.raises(expected_exception, *args, **kwargs)


# TODO: Consider moving these type utilities into pylibcudf.types itself.
def is_signed_integer(plc_dtype):
    return (
        plc.TypeId.INT8.value <= plc_dtype.id().value <= plc.TypeId.INT64.value
    )


def is_unsigned_integer(plc_dtype):
    return (
        plc.TypeId.UINT8.value
        <= plc_dtype.id().value
        <= plc.TypeId.UINT64.value
    )


def is_integer(plc_dtype):
    return is_signed_integer(plc_dtype) or is_unsigned_integer(plc_dtype)


def is_floating(plc_dtype):
    return (
        plc.TypeId.FLOAT32.value
        <= plc_dtype.id().value
        <= plc.TypeId.FLOAT64.value
    )


def is_boolean(plc_dtype):
    return plc_dtype.id() == plc.TypeId.BOOL8


def is_string(plc_dtype):
    return plc_dtype.id() == plc.TypeId.STRING


def is_fixed_width(plc_dtype):
    return (
        is_integer(plc_dtype)
        or is_floating(plc_dtype)
        or is_boolean(plc_dtype)
    )


def default_struct_testing_type(nullable=False):
    # We must explicitly specify this type via a field to ensure we don't include
    # nullability accidentally.
    return pa.struct([pa.field("v", pa.int64(), nullable=nullable)])
