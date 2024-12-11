# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_New,
    PyCapsule_SetName,
)
from libc.stdlib cimport free
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dataclasses import dataclass, field
from functools import singledispatch

from pyarrow import lib as pa

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.interop cimport (
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
    DLManagedTensor,
    column_metadata,
    from_arrow_column as cpp_from_arrow_column,
    from_arrow_stream as cpp_from_arrow_stream,
    from_dlpack as cpp_from_dlpack,
    to_arrow_host_raw,
    to_arrow_schema_raw,
    to_dlpack as cpp_to_dlpack,
)
from pylibcudf.libcudf.table.table cimport table

from . cimport copying
from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .types cimport DataType, type_id

__all__ = [
    "ColumnMetadata",
    "from_arrow",
    "from_dlpack",
    "to_arrow",
    "to_dlpack",
]

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
    raise TypeError(
        f"Unsupported type {type(pyarrow_object)} for conversion from arrow"
    )


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
        <ArrowArrayStream*>PyCapsule_GetPointer(stream, "arrow_array_stream")
    )

    cdef unique_ptr[table] c_result
    with nogil:
        # The libcudf function here will release the stream.
        c_result = cpp_from_arrow_stream(c_stream)

    return Table.from_libcudf(move(c_result))


@from_arrow.register(pa.Scalar)
def _from_arrow_scalar(pyarrow_object, *, DataType data_type=None):
    if isinstance(pyarrow_object.type, pa.ListType) and pyarrow_object.as_py() is None:
        # pyarrow doesn't correctly handle None values for list types, so
        # we have to create this one manually.
        # https://github.com/apache/arrow/issues/40319
        pa_array = pa.array([None], type=pyarrow_object.type)
    else:
        pa_array = pa.array([pyarrow_object])
    return copying.get_element(
        from_arrow(pa_array, data_type=data_type),
        0,
    )


@from_arrow.register(pa.Array)
def _from_arrow_column(pyarrow_object, *, DataType data_type=None):
    if data_type is not None:
        raise ValueError("data_type may not be passed for arrays")

    schema, array = pyarrow_object.__arrow_c_array__()
    cdef ArrowSchema* c_schema = (
        <ArrowSchema*>PyCapsule_GetPointer(schema, "arrow_schema")
    )
    cdef ArrowArray* c_array = (
        <ArrowArray*>PyCapsule_GetPointer(array, "arrow_array")
    )

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_from_arrow_column(c_schema, c_array)

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
    raise TypeError(f"Unsupported type {type(cudf_object)} for conversion to arrow")


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


cdef void _release_schema(object schema_capsule) noexcept:
    """Release the ArrowSchema object stored in a PyCapsule."""
    cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(
        schema_capsule, 'arrow_schema'
    )
    if schema.release != NULL:
        schema.release(schema)

    free(schema)


cdef void _release_array(object array_capsule) noexcept:
    """Release the ArrowArray object stored in a PyCapsule."""
    cdef ArrowArray* array = <ArrowArray*>PyCapsule_GetPointer(
        array_capsule, 'arrow_array'
    )
    if array.release != NULL:
        array.release(array)

    free(array)


def _table_to_schema(Table tbl, metadata):
    if metadata is None:
        metadata = [ColumnMetadata() for _ in range(len(tbl.columns()))]
    metadata = [ColumnMetadata(m) if isinstance(m, str) else m for m in metadata]

    cdef vector[column_metadata] c_metadata
    c_metadata.reserve(len(metadata))
    for meta in metadata:
        c_metadata.push_back(_metadata_to_libcudf(meta))

    cdef ArrowSchema* raw_schema_ptr
    with nogil:
        raw_schema_ptr = to_arrow_schema_raw(tbl.view(), c_metadata)

    return PyCapsule_New(<void*>raw_schema_ptr, 'arrow_schema', _release_schema)


def _table_to_host_array(Table tbl):
    cdef ArrowArray* raw_host_array_ptr
    with nogil:
        raw_host_array_ptr = to_arrow_host_raw(tbl.view())

    return PyCapsule_New(<void*>raw_host_array_ptr, "arrow_array", _release_array)


class _TableWithArrowMetadata:
    def __init__(self, tbl, metadata=None):
        self.tbl = tbl
        self.metadata = metadata

    def __arrow_c_array__(self, requested_schema=None):
        return _table_to_schema(self.tbl, self.metadata), _table_to_host_array(self.tbl)


# TODO: In the long run we should get rid of the `to_arrow` functions in favor of using
# the protocols directly via `pa.table(cudf_object, schema=...)` directly. We can do the
# same for columns. We cannot do this for scalars since there is no corresponding
# protocol. Since this will require broader changes throughout the codebase, the current
# approach is to leverage the protocol internally but to continue exposing `to_arrow`.
@to_arrow.register(Table)
def _to_arrow_table(cudf_object, metadata=None):
    test_table = _TableWithArrowMetadata(cudf_object, metadata)
    return pa.table(test_table)


@to_arrow.register(Column)
def _to_arrow_array(cudf_object, metadata=None):
    """Create a PyArrow array from a pylibcudf column."""
    if metadata is not None:
        metadata = [metadata]
    return to_arrow(Table([cudf_object]), metadata)[0]


@to_arrow.register(Scalar)
def _to_arrow_scalar(cudf_object, metadata=None):
    # Note that metadata for scalars is primarily important for preserving
    # information on nested types since names are otherwise irrelevant.
    return to_arrow(Column.from_scalar(cudf_object, 1), metadata=metadata)[0]


cpdef Table from_dlpack(object managed_tensor):
    """
    Convert a DLPack DLTensor into a cudf table.

    For details, see :cpp:func:`cudf::from_dlpack`

    Parameters
    ----------
    managed_tensor : PyCapsule
        A 1D or 2D column-major (Fortran order) tensor.

    Returns
    -------
    Table
        Table with a copy of the tensor data.
    """
    if not PyCapsule_IsValid(managed_tensor, "dltensor"):
        raise ValueError("Invalid PyCapsule object")
    cdef unique_ptr[table] c_result
    cdef DLManagedTensor* dlpack_tensor = <DLManagedTensor*>PyCapsule_GetPointer(
        managed_tensor, "dltensor"
    )
    if dlpack_tensor is NULL:
        raise ValueError("PyCapsule object contained a NULL pointer")
    PyCapsule_SetName(managed_tensor, "used_dltensor")

    # Note: A copy is always performed when converting the dlpack
    # data to a libcudf table. We also delete the dlpack_tensor pointer
    # as the pointer is not deleted by libcudf's from_dlpack function.
    # TODO: https://github.com/rapidsai/cudf/issues/10874
    # TODO: https://github.com/rapidsai/cudf/issues/10849
    with nogil:
        c_result = cpp_from_dlpack(dlpack_tensor)

    cdef Table result = Table.from_libcudf(move(c_result))
    dlpack_tensor.deleter(dlpack_tensor)
    return result


cpdef object to_dlpack(Table input):
    """
    Convert a cudf table into a DLPack DLTensor.

    For details, see :cpp:func:`cudf::to_dlpack`

    Parameters
    ----------
    input : Table
        A 1D or 2D column-major (Fortran order) tensor.

    Returns
    -------
    PyCapsule
        1D or 2D DLPack tensor with a copy of the table data, or nullptr.
    """
    for col in input._columns:
        if col.null_count():
            raise ValueError(
                "Cannot create a DLPack tensor with null values. "
                "Input is required to have null count as zero."
            )
    cdef DLManagedTensor *dlpack_tensor

    with nogil:
        dlpack_tensor = cpp_to_dlpack(input.view())

    return PyCapsule_New(
        dlpack_tensor,
        "dltensor",
        dlmanaged_tensor_pycapsule_deleter
    )


cdef void dlmanaged_tensor_pycapsule_deleter(object pycap_obj) noexcept:
    if PyCapsule_IsValid(pycap_obj, "used_dltensor"):
        # we do not call a used capsule's deleter
        return
    cdef DLManagedTensor* dlpack_tensor = <DLManagedTensor*>PyCapsule_GetPointer(
        pycap_obj, "dltensor"
    )
    if dlpack_tensor is not NULL:
        dlpack_tensor.deleter(dlpack_tensor)
