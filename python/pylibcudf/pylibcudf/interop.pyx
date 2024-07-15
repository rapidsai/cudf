# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cpython cimport pycapsule
from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pyarrow cimport lib as pa

from dataclasses import dataclass, field
from functools import singledispatch

from pyarrow import lib as pa

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.interop cimport (
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
    column_metadata,
    from_arrow as cpp_from_arrow,
    from_arrow_column as cpp_from_arrow_column,
    from_arrow_stream as cpp_from_arrow_stream,
    to_arrow as cpp_to_arrow,
)
from pylibcudf.libcudf.scalar.scalar cimport fixed_point_scalar, scalar
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.wrappers.decimals cimport (
    decimal32,
    decimal64,
    decimal128,
    scale_type,
)

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .types cimport DataType, type_id

ARROW_TO_PYLIBCUDF_TYPES = {
    pa.int8(): type_id.INT8,
    pa.int16(): type_id.INT16,
    pa.int32(): type_id.INT32,
    pa.int64(): type_id.INT64,
    pa.uint8(): type_id.UINT8,
    pa.uint16(): type_id.UINT16,
    pa.uint32(): type_id.UINT32,
    pa.uint64(): type_id.UINT64,
    pa.float32(): type_id.FLOAT32,
    pa.float64(): type_id.FLOAT64,
    pa.bool_(): type_id.BOOL8,
    pa.string(): type_id.STRING,
    pa.duration('s'): type_id.DURATION_SECONDS,
    pa.duration('ms'): type_id.DURATION_MILLISECONDS,
    pa.duration('us'): type_id.DURATION_MICROSECONDS,
    pa.duration('ns'): type_id.DURATION_NANOSECONDS,
    pa.timestamp('s'): type_id.TIMESTAMP_SECONDS,
    pa.timestamp('ms'): type_id.TIMESTAMP_MILLISECONDS,
    pa.timestamp('us'): type_id.TIMESTAMP_MICROSECONDS,
    pa.timestamp('ns'): type_id.TIMESTAMP_NANOSECONDS,
    pa.date32(): type_id.TIMESTAMP_DAYS,
    pa.null(): type_id.EMPTY,
}

LIBCUDF_TO_ARROW_TYPES = {
    v: k for k, v in ARROW_TO_PYLIBCUDF_TYPES.items()
}

cdef column_metadata _metadata_to_libcudf(metadata):
    """Convert a ColumnMetadata object to C++ column_metadata.

    Since this class is mutable and cheap, it is easier to create the C++
    object on the fly rather than have it directly backing the storage for
    the Cython class. Additionally, this structure restricts the dependency
    on C++ types to just within this module, allowing us to make the module a
    pure Python module (from an import sense, i.e. no pxd declarations).
    """
    cdef column_metadata c_metadata
    c_metadata.name = metadata.name.encode()
    for child_meta in metadata.children_meta:
        c_metadata.children_meta.push_back(_metadata_to_libcudf(child_meta))
    return c_metadata


@dataclass
class ColumnMetadata:
    """Metadata associated with a column.

    This is the Python representation of :cpp:class:`cudf::column_metadata`.
    """
    name: str = ""
    children_meta: list[ColumnMetadata] = field(default_factory=list)


@singledispatch
def from_arrow(pyarrow_object, *, DataType data_type=None):
    """Create a cudf object from a pyarrow object.

    Parameters
    ----------
    pyarrow_object : Union[pyarrow.Array, pyarrow.Table, pyarrow.Scalar]
        The PyArrow object to convert.

    Returns
    -------
    Union[Table, Scalar]
        The converted object of type corresponding to the input type in cudf.
    """
    raise TypeError("from_arrow only accepts Table and Scalar objects")


@from_arrow.register(pa.DataType)
def _from_arrow_datatype(pyarrow_object):
    if isinstance(pyarrow_object, pa.Decimal128Type):
        return DataType(type_id.DECIMAL128, scale=-pyarrow_object.scale)
    elif isinstance(pyarrow_object, pa.StructType):
        return DataType(type_id.STRUCT)
    elif isinstance(pyarrow_object, pa.ListType):
        return DataType(type_id.LIST)
    else:
        try:
            return DataType(ARROW_TO_PYLIBCUDF_TYPES[pyarrow_object])
        except KeyError:
            raise TypeError(f"Unable to convert {pyarrow_object} to cudf datatype")


@from_arrow.register(pa.Table)
def _from_arrow_table(pyarrow_object, *, DataType data_type=None):
    if data_type is not None:
        raise ValueError("data_type may not be passed for tables")
    stream = pyarrow_object.__arrow_c_stream__()
    cdef ArrowArrayStream* c_stream = (
        <ArrowArrayStream*>pycapsule.PyCapsule_GetPointer(stream, "arrow_array_stream")
    )

    cdef unique_ptr[table] c_result
    with nogil:
        # The libcudf function here will release the stream.
        c_result = move(cpp_from_arrow_stream(c_stream))

    return Table.from_libcudf(move(c_result))


@from_arrow.register(pa.Scalar)
def _from_arrow_scalar(pyarrow_object, *, DataType data_type=None):
    cdef shared_ptr[pa.CScalar] arrow_scalar = pa.pyarrow_unwrap_scalar(pyarrow_object)

    cdef unique_ptr[scalar] c_result
    with nogil:
        c_result = move(cpp_from_arrow(dereference(arrow_scalar)))

    cdef Scalar result = Scalar.from_libcudf(move(c_result))

    if result.type().id() != type_id.DECIMAL128:
        if data_type is not None:
            raise ValueError(
                "dtype may not be passed for non-decimal types"
            )
        return result

    if data_type is None:
        raise ValueError(
            "Decimal scalars must be constructed with a dtype"
        )

    cdef type_id tid = data_type.id()

    if tid == type_id.DECIMAL32:
        result.c_obj.reset(
            new fixed_point_scalar[decimal32](
                (
                    <fixed_point_scalar[decimal128]*> result.c_obj.get()
                ).value(),
                scale_type(-pyarrow_object.type.scale),
                result.c_obj.get().is_valid()
            )
        )
    elif tid == type_id.DECIMAL64:
        result.c_obj.reset(
            new fixed_point_scalar[decimal64](
                (
                    <fixed_point_scalar[decimal128]*> result.c_obj.get()
                ).value(),
                scale_type(-pyarrow_object.type.scale),
                result.c_obj.get().is_valid()
            )
        )
    elif tid != type_id.DECIMAL128:
        raise ValueError(
            "Decimal scalars may only be cast to decimals"
        )

    return result


@from_arrow.register(pa.Array)
@from_arrow.register(pa.ChunkedArray)
def _from_arrow_column(pyarrow_object, *, DataType data_type=None):
    if data_type is not None:
        raise ValueError("data_type may not be passed for arrays")

    schema, array = pyarrow_object.__arrow_c_array__()
    cdef ArrowSchema* c_schema = (
        <ArrowSchema*>pycapsule.PyCapsule_GetPointer(schema, "arrow_schema")
    )
    cdef ArrowArray* c_array = (
        <ArrowArray*>pycapsule.PyCapsule_GetPointer(array, "arrow_array")
    )

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_from_arrow_column(c_schema, c_array))

    # The capsule destructors should release automatically for us, but we
    # choose to do it explicitly here for clarity.
    c_schema.release(c_schema)
    c_array.release(c_array)

    return Column.from_libcudf(move(c_result))


@singledispatch
def to_arrow(cudf_object, metadata=None):
    """Convert to a PyArrow object.

    Parameters
    ----------
    cudf_object : Union[Column, Table, Scalar]
        The cudf object to convert.
    metadata : list
        The metadata to attach to the columns of the table.

    Returns
    -------
    Union[pyarrow.Array, pyarrow.Table, pyarrow.Scalar]
        The converted object of type corresponding to the input type in PyArrow.
    """
    raise TypeError("to_arrow only accepts Table and Scalar objects")


@to_arrow.register(DataType)
def _to_arrow_datatype(cudf_object, **kwargs):
    """
    Convert a datatype to arrow.

    Translation of some types requires extra information as a keyword
    argument. Specifically:

    - When translating a decimal type, provide ``precision``
    - When translating a struct type, provide ``fields``
    - When translating a list type, provide the wrapped ``value_type``
    """
    if cudf_object.id() in {type_id.DECIMAL32, type_id.DECIMAL64, type_id.DECIMAL128}:
        if not (precision := kwargs.get("precision")):
            raise ValueError(
                "Precision must be provided for decimal types"
            )
            # no pa.decimal32 or pa.decimal64
        return pa.decimal128(precision, -cudf_object.scale())
    elif cudf_object.id() == type_id.STRUCT:
        if not (fields := kwargs.get("fields")):
            raise ValueError(
                "Fields must be provided for struct types"
            )
        return pa.struct(fields)
    elif cudf_object.id() == type_id.LIST:
        if not (value_type := kwargs.get("value_type")):
            raise ValueError(
                "Value type must be provided for list types"
            )
        return pa.list_(value_type)
    else:
        try:
            return LIBCUDF_TO_ARROW_TYPES[cudf_object.id()]
        except KeyError:
            raise TypeError(
                f"Unable to convert {cudf_object.id()} to arrow datatype"
            )


@to_arrow.register(Table)
def _to_arrow_table(cudf_object, metadata=None):
    if metadata is None:
        metadata = [ColumnMetadata() for _ in range(len(cudf_object.columns()))]
    metadata = [ColumnMetadata(m) if isinstance(m, str) else m for m in metadata]
    cdef vector[column_metadata] c_table_metadata
    cdef shared_ptr[pa.CTable] c_table_result
    for meta in metadata:
        c_table_metadata.push_back(_metadata_to_libcudf(meta))
    with nogil:
        c_table_result = move(
            cpp_to_arrow((<Table> cudf_object).view(), c_table_metadata)
        )

    return pa.pyarrow_wrap_table(c_table_result)


@to_arrow.register(Scalar)
def _to_arrow_scalar(cudf_object, metadata=None):
    # Note that metadata for scalars is primarily important for preserving
    # information on nested types since names are otherwise irrelevant.
    if metadata is None:
        metadata = ColumnMetadata()
    metadata = ColumnMetadata(metadata) if isinstance(metadata, str) else metadata
    cdef column_metadata c_scalar_metadata = _metadata_to_libcudf(metadata)
    cdef shared_ptr[pa.CScalar] c_scalar_result
    with nogil:
        c_scalar_result = move(
            cpp_to_arrow(
                dereference((<Scalar> cudf_object).c_obj), c_scalar_metadata
            )
        )

    return pa.pyarrow_wrap_scalar(c_scalar_result)


@to_arrow.register(Column)
def _to_arrow_array(cudf_object, metadata=None):
    """Create a PyArrow array from a pylibcudf column."""
    if metadata is None:
        metadata = ColumnMetadata()
    metadata = ColumnMetadata(metadata) if isinstance(metadata, str) else metadata
    return to_arrow(Table([cudf_object]), [metadata])[0]
