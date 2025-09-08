# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_New,
    PyCapsule_SetName,
)
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from functools import singledispatch

from pylibcudf.libcudf.interop cimport (
    DLManagedTensor,
    from_dlpack as cpp_from_dlpack,
    to_dlpack as cpp_to_dlpack,
)
from pylibcudf.libcudf.table.table cimport table

from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .types cimport DataType, type_id
from .types import LIBCUDF_TO_ARROW_TYPES
from .utils cimport _get_stream
from ._interop_helpers import ColumnMetadata

try:
    import pyarrow as pa
    pa_err = None
except ImportError as e:
    pa = None
    pa_err = e


__all__ = [
    "ColumnMetadata",
    "from_arrow",
    "from_dlpack",
    "to_arrow",
    "to_dlpack",
]


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
    if pa_err is not None:
        raise RuntimeError(
            "pyarrow was not found on your system. Please "
            "pip install pylibcudf with the [pyarrow] extra for a "
            "compatible pyarrow version."
        ) from pa_err
    raise TypeError(
        f"Unsupported type {type(pyarrow_object)} for conversion from arrow"
    )


@singledispatch
def to_arrow(plc_object, metadata=None):
    """Convert to a PyArrow object.

    Parameters
    ----------
    plc_object : Union[Column, Table, Scalar]
        The cudf object to convert.
    metadata : list
        The metadata to attach to the columns of the table.

    Returns
    -------
    Union[pyarrow.Array, pyarrow.Table, pyarrow.Scalar]
        The converted object of type corresponding to the input type in PyArrow.
    """
    if pa_err is not None:
        raise RuntimeError(
            "pyarrow was not found on your system. Please "
            "pip install pylibcudf with the [pyarrow] extra for a "
            "compatible pyarrow version."
        ) from pa_err
    raise TypeError(f"Unsupported type {type(plc_object)} for conversion to arrow")


if pa is not None:
    @from_arrow.register(pa.DataType)
    def _from_arrow_datatype(pyarrow_object):
        return DataType.from_arrow(pyarrow_object)

    @from_arrow.register(pa.Table)
    def _from_arrow_table(pyarrow_object, *, DataType data_type=None):
        return Table.from_arrow(pyarrow_object, dtype=data_type)

    @from_arrow.register(pa.Scalar)
    def _from_arrow_scalar(pyarrow_object, *, DataType data_type=None):
        return Scalar.from_arrow(pyarrow_object, dtype=data_type)

    @from_arrow.register(pa.Array)
    def _from_arrow_column(pyarrow_object, *, DataType data_type=None):
        return Column.from_arrow(pyarrow_object, dtype=data_type)

    @to_arrow.register(DataType)
    def _to_arrow_datatype(plc_object, **kwargs):
        """
        Convert a datatype to arrow.

        Translation of some types requires extra information as a keyword
        argument. Specifically:

        - When translating a decimal type, provide ``precision``
        - When translating a struct type, provide ``fields``
        - When translating a list type, provide the wrapped ``value_type``
        """
        if plc_object.id() in {
            type_id.DECIMAL32,
            type_id.DECIMAL64,
            type_id.DECIMAL128
        }:
            if not (precision := kwargs.get("precision")):
                raise ValueError(
                    "Precision must be provided for decimal types"
                )
                # no pa.decimal32 or pa.decimal64
            return pa.decimal128(precision, -plc_object.scale())
        elif plc_object.id() == type_id.STRUCT:
            if not (fields := kwargs.get("fields")):
                raise ValueError(
                    "Fields must be provided for struct types"
                )
            return pa.struct(fields)
        elif plc_object.id() == type_id.LIST:
            if not (value_type := kwargs.get("value_type")):
                raise ValueError(
                    "Value type must be provided for list types"
                )
            return pa.list_(value_type)
        else:
            try:
                return LIBCUDF_TO_ARROW_TYPES[plc_object.id()]
            except KeyError:
                raise TypeError(
                    f"Unable to convert {plc_object.id()} to arrow datatype"
                )

    class _ObjectWithArrowMetadata:
        def __init__(self, obj, metadata=None):
            self.obj = obj
            self.metadata = metadata

        def __arrow_c_array__(self, requested_schema=None):
            return self.obj._to_schema(self.metadata), self.obj._to_host_array()

    @to_arrow.register(Table)
    def _to_arrow_table(plc_object, metadata=None):
        """Create a PyArrow table from a pylibcudf table."""
        return pa.table(_ObjectWithArrowMetadata(plc_object, metadata))

    @to_arrow.register(Column)
    def _to_arrow_array(plc_object, metadata=None):
        """Create a PyArrow array from a pylibcudf column."""
        return pa.array(_ObjectWithArrowMetadata(plc_object, metadata))

    @to_arrow.register(Scalar)
    def _to_arrow_scalar(plc_object, metadata=None):
        # Note that metadata for scalars is primarily important for preserving
        # information on nested types since names are otherwise irrelevant.
        return to_arrow(Column.from_scalar(plc_object, 1), metadata=metadata)[0]


cpdef Table from_dlpack(object managed_tensor, Stream stream=None):
    """
    Convert a DLPack DLTensor into a cudf table.

    For details, see :cpp:func:`cudf::from_dlpack`

    Parameters
    ----------
    managed_tensor : PyCapsule
        A 1D or 2D column-major (Fortran order) tensor.
    stream : Stream | None
        CUDA stream on which to perform the operation.

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
    stream = _get_stream(stream)

    # Note: A copy is always performed when converting the dlpack
    # data to a libcudf table. We also delete the dlpack_tensor pointer
    # as the pointer is not deleted by libcudf's from_dlpack function.
    # TODO: https://github.com/rapidsai/cudf/issues/10874
    # TODO: https://github.com/rapidsai/cudf/issues/10849
    with nogil:
        c_result = cpp_from_dlpack(dlpack_tensor, stream.view())

    cdef Table result = Table.from_libcudf(move(c_result), stream)
    dlpack_tensor.deleter(dlpack_tensor)
    return result


cpdef object to_dlpack(Table input, Stream stream=None):
    """
    Convert a cudf table into a DLPack DLTensor.

    For details, see :cpp:func:`cudf::to_dlpack`

    Parameters
    ----------
    input : Table
        A 1D or 2D column-major (Fortran order) tensor.
    stream : Stream | None
        CUDA stream on which to perform the operation.

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
    stream = _get_stream(stream)

    with nogil:
        dlpack_tensor = cpp_to_dlpack(input.view(), stream.view())

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
