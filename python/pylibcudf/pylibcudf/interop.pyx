# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_New,
    PyCapsule_SetName,
)
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from functools import singledispatch
import warnings

from pylibcudf.libcudf.interop cimport (
    DLManagedTensor,
    from_dlpack as cpp_from_dlpack,
    to_dlpack as cpp_to_dlpack,
)
from pylibcudf.libcudf.table.table cimport table

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .types cimport DataType
from .utils cimport _get_stream, _get_memory_resource
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


def _deprecated_to_arrow_warning():
    warnings.warn(
        "pylibcudf.interop.to_arrow is deprecated; call the object's .to_arrow(...) "
        "method instead (e.g., Table.to_arrow, Column.to_arrow, etc.).",
        FutureWarning,
        stacklevel=2,
    )


def _deprecated_from_arrow_warning():
    warnings.warn(
        "pylibcudf.interop.from_arrow is deprecated; use class methods instead "
        "(e.g., Table.from_arrow, Column.from_arrow, etc.).",
        FutureWarning,
        stacklevel=2,
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
    if pa_err is not None:
        raise RuntimeError(
            "pyarrow was not found on your system. Please "
            "pip install pylibcudf with the [pyarrow] extra for a "
            "compatible pyarrow version."
        ) from pa_err
    _deprecated_from_arrow_warning()
    raise TypeError(
        f"Unsupported type {type(pyarrow_object)} for conversion from arrow"
    )


@singledispatch
def to_arrow(plc_object, **kwargs):
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
    _deprecated_to_arrow_warning()
    raise TypeError(f"Unsupported type {type(plc_object)} for conversion to arrow")


if pa is not None:
    @from_arrow.register(pa.DataType)
    def _from_arrow_datatype(pyarrow_object):
        _deprecated_from_arrow_warning()
        return DataType.from_arrow(pyarrow_object)

    @from_arrow.register(pa.Table)
    def _from_arrow_table(pyarrow_object, *, DataType data_type=None):
        _deprecated_from_arrow_warning()
        return Table.from_arrow(pyarrow_object, dtype=data_type)

    @from_arrow.register(pa.Scalar)
    def _from_arrow_scalar(
        pyarrow_object,
        *,
        DataType data_type=None,
        Stream stream = None
    ):
        _deprecated_from_arrow_warning()
        return Scalar.from_arrow(pyarrow_object, dtype=data_type, stream=stream)

    @from_arrow.register(pa.Array)
    def _from_arrow_column(pyarrow_object, *, DataType data_type=None):
        _deprecated_from_arrow_warning()
        return Column.from_arrow(pyarrow_object, dtype=data_type)

    @to_arrow.register(DataType)
    def _to_arrow_datatype(plc_object, **kwargs):
        """Convert a datatype to arrow."""
        _deprecated_to_arrow_warning()
        return plc_object.to_arrow(**kwargs)

    @to_arrow.register(Table)
    def _to_arrow_table(plc_object, metadata=None):
        """Create a PyArrow table from a pylibcudf table."""
        _deprecated_to_arrow_warning()
        return plc_object.to_arrow(metadata=metadata)

    @to_arrow.register(Column)
    def _to_arrow_array(plc_object, metadata=None):
        """Create a PyArrow array from a pylibcudf column."""
        _deprecated_to_arrow_warning()
        return plc_object.to_arrow(metadata=metadata)

    @to_arrow.register(Scalar)
    def _to_arrow_scalar(plc_object, metadata=None):
        _deprecated_to_arrow_warning()
        return plc_object.to_arrow(metadata=metadata)


cpdef Table from_dlpack(
    object managed_tensor, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Convert a DLPack DLTensor into a cudf table.

    For details, see :cpp:func:`from_dlpack`

    Parameters
    ----------
    managed_tensor : PyCapsule
        A 1D or 2D column-major (Fortran order) tensor.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

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
    mr = _get_memory_resource(mr)

    # Note: A copy is always performed when converting the dlpack
    # data to a libcudf table. We also delete the dlpack_tensor pointer
    # as the pointer is not deleted by libcudf's from_dlpack function.
    # TODO: https://github.com/rapidsai/cudf/issues/10874
    # TODO: https://github.com/rapidsai/cudf/issues/10849
    with nogil:
        c_result = cpp_from_dlpack(dlpack_tensor, stream.view(), mr.get_mr())

    cdef Table result = Table.from_libcudf(move(c_result), stream, mr)
    dlpack_tensor.deleter(dlpack_tensor)
    return result


cpdef object to_dlpack(Table input, Stream stream=None, DeviceMemoryResource mr=None):
    """
    Convert a cudf table into a DLPack DLTensor.

    For details, see :cpp:func:`to_dlpack`

    Parameters
    ----------
    input : Table
        A 1D or 2D column-major (Fortran order) tensor.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned DLPack tensor's device
        memory.

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
    mr = _get_memory_resource(mr)

    with nogil:
        dlpack_tensor = cpp_to_dlpack(input.view(), stream.view(), mr.get_mr())

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
