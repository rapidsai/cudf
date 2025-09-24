# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_New,
    PyCapsule_SetName,
)
from cpython.ref cimport Py_DECREF, Py_INCREF

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from functools import singledispatch

from rmm.librmm cimport cuda_stream_view

from pylibcudf.libcudf.utilities.default_stream cimport get_default_stream
from pylibcudf.libcudf.interop cimport (
    DLManagedTensor,
    DLManagedTensorVersioned,
    from_dlpack as cpp_from_dlpack,
    to_dlpack as cpp_to_dlpack,
    to_dlpack_versioned as cpp_to_dlpack_versioned,
)
from pylibcudf.libcudf.table.table cimport table

from rmm.pylibrmm.stream cimport Stream
from rmm.librmm.device_buffer cimport get_current_cuda_device
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
    "get_dlpack_device",
    "to_dlpack_col",
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
    raise TypeError(f"Unsupported type {type(plc_object)} for conversion to arrow")


if pa is not None:
    @from_arrow.register(pa.DataType)
    def _from_arrow_datatype(pyarrow_object):
        return DataType.from_arrow(pyarrow_object)

    @from_arrow.register(pa.Table)
    def _from_arrow_table(pyarrow_object, *, DataType data_type=None):
        return Table.from_arrow(pyarrow_object, dtype=data_type)

    @from_arrow.register(pa.Scalar)
    def _from_arrow_scalar(
        pyarrow_object,
        *,
        DataType data_type=None,
        Stream stream = None
    ):
        return Scalar.from_arrow(pyarrow_object, dtype=data_type, stream=stream)

    @from_arrow.register(pa.Array)
    def _from_arrow_column(pyarrow_object, *, DataType data_type=None):
        return Column.from_arrow(pyarrow_object, dtype=data_type)

    @to_arrow.register(DataType)
    def _to_arrow_datatype(plc_object, **kwargs):
        """Convert a datatype to arrow."""
        return plc_object.to_arrow(**kwargs)

    @to_arrow.register(Table)
    def _to_arrow_table(plc_object, metadata=None):
        """Create a PyArrow table from a pylibcudf table."""
        return plc_object.to_arrow(metadata=metadata)

    @to_arrow.register(Column)
    def _to_arrow_array(plc_object, metadata=None):
        """Create a PyArrow array from a pylibcudf column."""
        return plc_object.to_arrow(metadata=metadata)

    @to_arrow.register(Scalar)
    def _to_arrow_scalar(plc_object, metadata=None):
        return plc_object.to_arrow(metadata=metadata)


cpdef Table from_dlpack(
    object managed_tensor, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Convert a DLPack DLTensor into a cudf table.

    For details, see :cpp:func:`cudf::from_dlpack`

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

    For details, see :cpp:func:`cudf::to_dlpack`

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


cdef void _dltensor_delete_owner(void *ctx) noexcept nogil:
    """Helper to delete the column owning the dlpack tensor data."""
    with gil:
        Py_DECREF(<object>ctx)


cpdef _get_dlpack_device():
    # Indicate as CUDA memory (although it could be pinned in theory).
    return (2, get_current_cuda_device().value())


cpdef object to_dlpack_col(
    Column self,
    stream=None,
    max_version=None,
    dl_device=None,
    copy=None,
):
    cdef DLManagedTensorVersioned* dlmtensorv
    cdef bint to_cpu = False
    cdef cuda_stream_view.cuda_stream_view c_stream

    if max_version is None or max_version[0] == 0:
        raise BufferError("Unable to export DLPack with version <1.0.")

    if dl_device is not None:
        if dl_device == (1, 0):
            to_cpu = True
            if copy is None:
                copy = True  # moving to CPU is always a copy
            elif not copy:
                raise BufferError("copying to CPU requires a copy.")

        elif dl_device != self.__dlpack_device__:
            raise BufferError("cudf only supports dl_device if identical or CPU.")

    if self.null_count() != 0:
        # A BufferError is nicer, so explicitly check in Python
        # (we could convert a custom exception from C->Cython also)
        raise BufferError("Cannot export column with nulls.")

    if copy is None:
        copy = False

    if to_cpu:
        # just ignore stream, it should match the device, cpu has none
        # (could guess it's a GPU one, but that isn't strictly standardized)
        c_stream = get_default_stream()
    elif stream == -1:
        # Passing the default stream means we don't do any synchronization
        # (so use our stream which enforces no order between streams).
        c_stream = get_default_stream()
    elif stream is None or stream == 1:
        c_stream = cuda_stream_view.cuda_stream_legacy
    elif stream == 2:
        c_stream = cuda_stream_view.cuda_stream_default
    elif not isinstance(stream, int) or stream < 3:
        raise ValueError(
            f'On CUDA, the valid stream for the DLPack protocol is -1,'
            f' 1, 2, or any larger value, but {stream} was provided')
    else:
        # User provided a custom stream.
        c_stream = cuda_stream_view.cuda_stream_view(
            <cuda_stream_view.cudaStream_t><uintptr_t>stream)

    dlmtensorv = cpp_to_dlpack_versioned(
        self.view(), copy, to_cpu, c_stream, _dltensor_delete_owner, <void *>self)
    Py_INCREF(self)  # on success, dlmtensorv takes a reference to self
    try:
        capsule = PyCapsule_New(
            dlmtensorv, 'dltensor_versioned', dlmanaged_tensor_pycapsule_deleter)
    except BaseException:  # ownership not transferred to capsule
        dlmtensorv.deleter(dlmtensorv)
        raise

    if to_cpu:
        # Make sure data is CPU available
        c_stream.synchronize()

    return capsule


cdef void dlmanaged_tensor_pycapsule_deleter(object capsule) noexcept:
    """Delete the dlpack tensor stored inside the capsule (if needed)."""
    cdef DLManagedTensorVersioned *dltensorv

    if PyCapsule_IsValid(capsule, "dltensor_versioned"):
        dltensorv = <DLManagedTensorVersioned *>PyCapsule_GetPointer(
            capsule, "dltensor_versioned")
        dltensorv.deleter(dltensorv)

    # If the capsule was renamed, the tensor is already free'd.
