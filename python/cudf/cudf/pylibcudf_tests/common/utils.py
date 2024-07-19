# Copyright (c) 2024, NVIDIA CORPORATION.
from __future__ import annotations

import io
import os

import numpy as np
import pyarrow as pa
import pytest

from cudf._lib import pylibcudf as plc
from cudf._lib.pylibcudf.io.types import CompressionType


def metadata_from_arrow_type(
    pa_type: pa.Array,
    name: str = "",
) -> plc.interop.ColumnMetadata | None:
    metadata = plc.interop.ColumnMetadata(name)
    if pa.types.is_list(pa_type):
        child_meta = [plc.interop.ColumnMetadata("offsets")]
        for i in range(pa_type.num_fields):
            field_meta = metadata_from_arrow_type(
                pa_type.field(i).type, pa_type.field(i).name
            )
            child_meta.append(field_meta)
        metadata = plc.interop.ColumnMetadata(name, child_meta)
    elif pa.types.is_struct(pa_type):
        child_meta = []
        for i in range(pa_type.num_fields):
            field_meta = metadata_from_arrow_type(
                pa_type.field(i).type, pa_type.field(i).name
            )
            child_meta.append(field_meta)
        metadata = plc.interop.ColumnMetadata(
            name,
            # libcudf does not store field names, so just match pyarrow's.
            child_meta,
        )
    return metadata


def assert_column_eq(
    lhs: pa.Array | plc.Column,
    rhs: pa.Array | plc.Column,
    check_field_nullability=True,
) -> None:
    """Verify that a pylibcudf array and PyArrow array are equal.

    Parameters
    ----------
    lhs: Union[pa.Array, plc.Column]
        The array with the expected values
    rhs: Union[pa.Array, plc.Column]
        The array to check
    check_field_nullability:
        For list/struct dtypes, whether to check if the nullable attributes
        on child fields are equal.

        Useful for checking roundtripping of lossy formats like JSON that may not
        preserve this information.
    """
    # Nested types require children metadata to be passed to the conversion function.
    if isinstance(lhs, (pa.Array, pa.ChunkedArray)) and isinstance(
        rhs, plc.Column
    ):
        rhs = plc.interop.to_arrow(
            rhs, metadata=metadata_from_arrow_type(lhs.type)
        )
    elif isinstance(lhs, plc.Column) and isinstance(
        rhs, (pa.Array, pa.ChunkedArray)
    ):
        lhs = plc.interop.to_arrow(
            lhs, metadata=metadata_from_arrow_type(rhs.type)
        )
    else:
        raise ValueError(
            "One of the inputs must be a Column and the other an Array"
        )

    if isinstance(lhs, pa.ChunkedArray):
        lhs = lhs.combine_chunks()
    if isinstance(rhs, pa.ChunkedArray):
        rhs = rhs.combine_chunks()

    def _make_fields_nullable(typ):
        new_fields = []
        for i in range(typ.num_fields):
            child_field = typ.field(i)
            if not child_field.nullable:
                child_type = child_field.type
                if isinstance(child_field.type, (pa.StructType, pa.ListType)):
                    child_type = _make_fields_nullable(child_type)
                new_fields.append(
                    pa.field(child_field.name, child_type, nullable=True)
                )
            else:
                new_fields.append(child_field)

        if isinstance(typ, pa.StructType):
            return pa.struct(new_fields)
        elif isinstance(typ, pa.ListType):
            return pa.list_(new_fields[0])
        return typ

    if not check_field_nullability:
        rhs_type = _make_fields_nullable(rhs.type)
        rhs = rhs.cast(rhs_type)

        lhs_type = _make_fields_nullable(lhs.type)
        lhs = rhs.cast(lhs_type)

    if pa.types.is_floating(lhs.type) and pa.types.is_floating(rhs.type):
        lhs_nans = pa.compute.is_nan(lhs)
        rhs_nans = pa.compute.is_nan(rhs)
        assert lhs_nans.equals(rhs_nans)

        if pa.compute.any(lhs_nans) or pa.compute.any(rhs_nans):
            # masks must be equal at this point
            mask = pa.compute.fill_null(pa.compute.invert(lhs_nans), True)
            lhs = lhs.filter(mask)
            rhs = rhs.filter(mask)

        np.testing.assert_array_almost_equal(lhs, rhs)
    else:
        assert lhs.equals(rhs)


def assert_table_eq(pa_table: pa.Table, plc_table: plc.Table) -> None:
    """Verify that a pylibcudf table and PyArrow table are equal."""
    plc_shape = (plc_table.num_rows(), plc_table.num_columns())
    assert plc_shape == pa_table.shape

    for plc_col, pa_col in zip(plc_table.columns(), pa_table.columns):
        assert_column_eq(pa_col, plc_col)


def assert_table_and_meta_eq(
    pa_table: pa.Table,
    plc_table_w_meta: plc.io.types.TableWithMetadata,
    check_field_nullability=True,
    check_types_if_empty=True,
    check_names=True,
) -> None:
    """Verify that the pylibcudf TableWithMetadata and PyArrow table are equal"""

    plc_table = plc_table_w_meta.tbl

    plc_shape = (plc_table.num_rows(), plc_table.num_columns())
    assert (
        plc_shape == pa_table.shape
    ), f"{plc_shape} is not equal to {pa_table.shape}"

    if not check_types_if_empty and plc_table.num_rows() == 0:
        return

    for plc_col, pa_col in zip(plc_table.columns(), pa_table.columns):
        assert_column_eq(pa_col, plc_col, check_field_nullability)

    # Check column name equality
    if check_names:
        assert (
            plc_table_w_meta.column_names() == pa_table.column_names
        ), f"{plc_table_w_meta.column_names()} != {pa_table.column_names}"


def cudf_raises(expected_exception: BaseException, *args, **kwargs):
    # A simple wrapper around pytest.raises that defaults to looking for cudf exceptions
    match = kwargs.get("match", None)
    if match is None:
        kwargs["match"] = "CUDF failure at"
    return pytest.raises(expected_exception, *args, **kwargs)


def is_string(plc_dtype: plc.DataType):
    return plc_dtype.id() == plc.TypeId.STRING


def nesting_level(typ) -> tuple[int, int]:
    """Return list and struct nesting of a pyarrow type."""
    if isinstance(typ, pa.ListType):
        list_, struct = nesting_level(typ.value_type)
        return list_ + 1, struct
    elif isinstance(typ, pa.StructType):
        lists, structs = map(max, zip(*(nesting_level(t.type) for t in typ)))
        return lists, structs + 1
    else:
        return 0, 0


def is_nested_struct(typ):
    return nesting_level(typ)[1] > 1


def is_nested_list(typ):
    return nesting_level(typ)[0] > 1


def _convert_numeric_types_to_floating(pa_table):
    """
    Useful little helper for testing the
    dtypes option in I/O readers.

    Returns a tuple containing the pylibcudf dtypes
    and the new pyarrow schema
    """
    dtypes = []
    new_fields = []
    for i in range(len(pa_table.schema)):
        field = pa_table.schema.field(i)
        child_types = []

        plc_type = plc.interop.from_arrow(field.type)
        if pa.types.is_integer(field.type) or pa.types.is_unsigned_integer(
            field.type
        ):
            plc_type = plc.interop.from_arrow(pa.float64())
            field = field.with_type(pa.float64())

        dtypes.append((field.name, plc_type, child_types))

        new_fields.append(field)
    return dtypes, new_fields


def write_source_str(source, input_str):
    """
    Write a string to the source
    (useful for testing CSV/JSON I/O)
    """
    if not isinstance(source, io.IOBase):
        with open(source, "w") as source_f:
            source_f.write(input_str)
    else:
        if isinstance(source, io.BytesIO):
            input_str = input_str.encode("utf-8")
        source.write(input_str)
        source.seek(0)


def sink_to_str(sink):
    """
    Takes a sink (e.g. StringIO/BytesIO, filepath, etc.)
    and reads in the contents into a string (str not bytes)
    for comparison
    """
    if isinstance(sink, (str, os.PathLike)):
        with open(sink, "r") as f:
            str_result = f.read()
    elif isinstance(sink, io.BytesIO):
        sink.seek(0)
        str_result = sink.read().decode()
    else:
        sink.seek(0)
        str_result = sink.read()
    return str_result


def make_source(path_or_buf, pa_table, format, **kwargs):
    """
    Write a pyarrow Table to a specific format using pandas
    by dispatching to the appropriate to_* call.
    The caller is responsible for making sure that no arguments
    unsupported by pandas are passed in.
    """
    df = pa_table.to_pandas()
    mode = "w"
    if "compression" in kwargs:
        kwargs["compression"] = COMPRESSION_TYPE_TO_PANDAS[
            kwargs["compression"]
        ]
        if kwargs["compression"] is not None and format != "json":
            # pandas json method only supports mode="w"/"a"
            mode = "wb"
    if format == "json":
        df.to_json(path_or_buf, mode=mode, **kwargs)
    elif format == "csv":
        df.to_csv(path_or_buf, mode=mode, **kwargs)
    if isinstance(path_or_buf, io.IOBase):
        path_or_buf.seek(0)
    return path_or_buf


NUMERIC_PA_TYPES = [pa.int64(), pa.float64(), pa.uint64()]
STRING_PA_TYPES = [pa.string()]
BOOL_PA_TYPES = [pa.bool_()]
LIST_PA_TYPES = [
    pa.list_(pa.int64()),
    # Nested case
    pa.list_(pa.list_(pa.int64())),
]

# We must explicitly specify this type via a field to ensure we don't include
# nullability accidentally.
DEFAULT_STRUCT_TESTING_TYPE = pa.struct(
    [pa.field("v", pa.int64(), nullable=False)]
)
NESTED_STRUCT_TESTING_TYPE = pa.struct(
    [
        pa.field("a", pa.int64(), nullable=False),
        pa.field(
            "b_struct",
            pa.struct([pa.field("b", pa.float64(), nullable=False)]),
            nullable=False,
        ),
    ]
)

DEFAULT_PA_STRUCT_TESTING_TYPES = [
    DEFAULT_STRUCT_TESTING_TYPE,
    NESTED_STRUCT_TESTING_TYPE,
]

DEFAULT_PA_TYPES = (
    NUMERIC_PA_TYPES
    + STRING_PA_TYPES
    + BOOL_PA_TYPES
    + LIST_PA_TYPES
    + DEFAULT_PA_STRUCT_TESTING_TYPES
)

# Map pylibcudf compression types to pandas ones
# Not all compression types map cleanly, read the comments to learn more!
# If a compression type is unsupported, it maps to False.

COMPRESSION_TYPE_TO_PANDAS = {
    CompressionType.NONE: None,
    # Users of this dict will have to special case
    # AUTO
    CompressionType.AUTO: None,
    CompressionType.GZIP: "gzip",
    CompressionType.BZIP2: "bz2",
    CompressionType.ZIP: "zip",
    CompressionType.XZ: "xz",
    CompressionType.ZSTD: "zstd",
    # Unsupported
    CompressionType.ZLIB: False,
    CompressionType.LZ4: False,
    CompressionType.LZO: False,
    # These only work for parquet
    CompressionType.SNAPPY: "snappy",
    CompressionType.BROTLI: "brotli",
}
ALL_PA_TYPES = DEFAULT_PA_TYPES
