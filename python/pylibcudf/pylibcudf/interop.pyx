# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_New,
    PyCapsule_SetName,
)
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from dataclasses import dataclass, field
from functools import singledispatch

from pyarrow import lib as pa

from pylibcudf.libcudf.interop cimport (
    DLManagedTensor,
    from_dlpack as cpp_from_dlpack,
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


@dataclass
class ColumnMetadata:
    """Metadata associated with a column.

    This is the Python representation of :cpp:class:`cudf::column_metadata`.
    """
    name: str = ""
    children_meta: list[ColumnMetadata] = field(default_factory=list)


def _create_nested_column_metadata(Column col):
    return ColumnMetadata(
        children_meta=[
            _create_nested_column_metadata(child) for child in col.children()
        ]
    )


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
    return Table(pyarrow_object)


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

    return Column(pyarrow_object)


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


class _ObjectWithArrowMetadata:
    def __init__(self, obj, metadata=None):
        self.obj = obj
        self.metadata = metadata

    def __arrow_c_array__(self, requested_schema=None):
        return self.obj._to_schema(self.metadata), self.obj._to_host_array()


@to_arrow.register(Table)
def _to_arrow_table(cudf_object, metadata=None):
    """Create a PyArrow table from a pylibcudf table."""
    # TODO: See if we can stop supporting configuration of struct field names when
    # exporting to arrow data. That would allow us to get rid of the
    # _ObjectWithArrowMetadata struct and just use the underlying Table directly.
    return pa.table(_ObjectWithArrowMetadata(cudf_object, metadata))


@to_arrow.register(Column)
def _to_arrow_array(cudf_object, metadata=None):
    """Create a PyArrow array from a pylibcudf column."""
    return pa.array(_ObjectWithArrowMetadata(cudf_object, metadata))


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
