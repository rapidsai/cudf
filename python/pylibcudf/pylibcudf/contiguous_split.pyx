# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libc.stdint cimport uint8_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.contiguous_split cimport (
    pack as cpp_pack,
    packed_columns,
    unpack as cpp_unpack,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view

from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from .gpumemoryview cimport gpumemoryview
from .table cimport Table
from .utils cimport int_to_void_ptr


__all__ = [
    "PackedColumns",
    "pack",
    "unpack",
    "unpack_from_memoryviews",
]

cdef class HostBuffer:
    """Owning host buffer that implements the buffer protocol"""
    @staticmethod
    cdef HostBuffer from_unique_ptr(
        unique_ptr[vector[uint8_t]] vec
    ):
        cdef HostBuffer out = HostBuffer.__new__(HostBuffer)
        # Allow construction from nullptr
        out.nbytes = 0 if vec.get() == NULL else dereference(vec).size()
        out.c_obj = move(vec)
        out.shape[0] = out.nbytes
        out.strides[0] = 1
        return out

    __hash__ = None

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # Empty vec produces empty buffer
        buffer.buf = NULL if self.nbytes == 0 else dereference(self.c_obj).data()
        buffer.format = NULL  # byte
        buffer.internal = NULL
        buffer.itemsize = 1
        buffer.len = self.nbytes
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass


cdef class PackedColumns:
    """Column data in a serialized format.

    Contains data from an array of columns in two contiguous buffers:
    one on host, which contains table metadata and one on device which
    contains the table data.

    For details, see :cpp:class:`cudf::packed_columns`.
    """
    def __init__(self):
        raise ValueError(
            "PackedColumns should not be constructed directly. "
            "Use one of the factories."
        )

    __hash__ = None

    @staticmethod
    cdef PackedColumns from_libcudf(unique_ptr[packed_columns] data):
        """Create a Python PackedColumns from a libcudf packed_columns."""
        cdef PackedColumns out = PackedColumns.__new__(PackedColumns)
        out.c_obj = move(data)
        return out

    cpdef tuple release(self):
        """Releases and returns the underlying serialized metadata and gpu data.

        The ownership of the memory are transferred to the returned buffers. After
        this call, `self` is empty.

        Returns
        -------
        memoryview (of a HostBuffer)
            The serialized metadata as contiguous host memory.
        gpumemoryview (of a rmm.DeviceBuffer)
            The serialized gpu data as contiguous device memory.
        """
        if not (dereference(self.c_obj).metadata and dereference(self.c_obj).gpu_data):
            raise ValueError("Cannot release empty PackedColumns")

        return (
            memoryview(
                HostBuffer.from_unique_ptr(move(dereference(self.c_obj).metadata))
            ),
            gpumemoryview(
                DeviceBuffer.c_from_unique_ptr(move(dereference(self.c_obj).gpu_data))
            )
        )


cpdef PackedColumns pack(Table input):
    """Deep-copy a table into a serialized contiguous memory format.

    Later use `unpack` or `unpack_from_memoryviews` to unpack the serialized
    data back into the table.

    Examples
    --------
    >>> packed = pylibcudf.contiguous_split.pack(...)
    >>> # Either unpack the whole `PackedColumns` at once.
    >>> pylibcudf.contiguous_split.unpack(packed)
    >>> # Or unpack the two serialized buffers in `PackedColumns`.
    >>> metadata, gpu_data = packed.release()
    >>> pylibcudf.contiguous_split.unpack_from_memoryviews(metadata, gpu_data)

    For details, see :cpp:func:`cudf::pack`.

    Parameters
    ----------
    input : Table
        Table to pack.

    Returns
    -------
    PackedColumns
        The packed columns.
    """
    return PackedColumns.from_libcudf(
        make_unique[packed_columns](cpp_pack(input.view()))
    )


cpdef Table unpack(PackedColumns input):
    """Deserialize the result of `pack`.

    Copies the result of a serialized table into a table.
    Contrary to the libcudf C++ function, the returned table is a copy
    of the serialized data.

    For details, see :cpp:func:`cudf::unpack`.

    Parameters
    ----------
    input : PackedColumns
        The packed columns to unpack.

    Returns
    -------
    Table
        Copy of the packed columns.
    """
    cdef table_view v = cpp_unpack(dereference(input.c_obj))
    # Since `Table.from_table_view` doesn't support an arbitrary owning object,
    # we copy the table, see <https://github.com/rapidsai/cudf/issues/17040>.
    cdef unique_ptr[table] t = make_unique[table](v)
    return Table.from_libcudf(move(t))


cpdef Table unpack_from_memoryviews(memoryview metadata, gpumemoryview gpu_data):
    """Deserialize the result of `pack`.

    Copies the result of a serialized table into a table.
    Contrary to the libcudf C++ function, the returned table is a copy
    of the serialized data.

    For details, see :cpp:func:`cudf::unpack`.

    Parameters
    ----------
    metadata : memoryview
        The packed metadata to unpack.
    gpu_data : gpumemoryview
        The packed gpu_data to unpack.

    Returns
    -------
    Table
        Copy of the packed columns.
    """
    if metadata.nbytes == 0:
        if gpu_data.__cuda_array_interface__["data"][0] != 0:
            raise ValueError("Expected an empty gpu_data from unpacking an empty table")
        return Table.from_libcudf(make_unique[table](table_view()))

    # Extract the raw data pointers
    cdef const uint8_t[::1] _metadata = metadata
    cdef const uint8_t* metadata_ptr = &_metadata[0]
    cdef const uint8_t* gpu_data_ptr = <uint8_t*>int_to_void_ptr(gpu_data.ptr)

    cdef table_view v = cpp_unpack(metadata_ptr, gpu_data_ptr)
    # Since `Table.from_table_view` doesn't support an arbitrary owning object,
    # we copy the table, see <https://github.com/rapidsai/cudf/issues/17040>.
    cdef unique_ptr[table] t = make_unique[table](v)
    return Table.from_libcudf(move(t))
