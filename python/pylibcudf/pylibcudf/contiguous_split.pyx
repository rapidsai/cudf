# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cuda.bindings.cyruntime cimport (
    cudaError,
    cudaError_t,
    cudaMemcpyAsync,
    cudaMemcpyKind,
)

from pylibcudf.libcudf.contiguous_split cimport (
    chunked_pack,
    pack as cpp_pack,
    packed_columns,
    unpack as cpp_unpack,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.utilities.span cimport device_span

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .gpumemoryview cimport gpumemoryview
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource


__all__ = [
    "ChunkedPack",
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
    cdef PackedColumns from_libcudf(
        unique_ptr[packed_columns] data,
        Stream stream,
        DeviceMemoryResource mr
    ):
        """Create a Python PackedColumns from a libcudf packed_columns."""
        cdef PackedColumns out = PackedColumns.__new__(PackedColumns)
        out.c_obj = move(data)
        out.stream = stream
        out.mr = mr
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
                DeviceBuffer.c_from_unique_ptr(
                    move(dereference(self.c_obj).gpu_data),
                    self.stream,
                    self.mr
                )
            )
        )


cdef class ChunkedPack:
    """
    A chunked version of :func:`pack`.

    This object can be used to pack (and therefore serialize) a table
    piece-by-piece through a user-provided staging buffer. This is
    useful when we want the end result to end up in host memory, but
    want control over the memory footprint.
    """
    def __init__(self):
        raise ValueError(
            "ChunkedPack should not be constructed directly. Use create instead."
        )

    @staticmethod
    def create(
        Table input,
        size_t user_buffer_size,
        Stream stream,
        DeviceMemoryResource temp_mr=None,
    ):
        """
        Create a chunked packer.

        Parameters
        ----------
        input
            The table to pack.
        user_buffer_size
            Size of the staging buffer to pack into, must be at least 1MB.
        stream : Stream | None
            Stream used for device memory operations and kernel launches.
        temp_mr : DeviceMemoryResource | None
            Memory resource for scratch allocations.

        Returns
        -------
        New ChunkedPack object.
        """
        temp_mr = _get_memory_resource(temp_mr)
        cdef unique_ptr[chunked_pack] obj = chunked_pack.create(
            input.view(), user_buffer_size, stream.view(), temp_mr.get_mr()
        )

        cdef ChunkedPack out = ChunkedPack.__new__(ChunkedPack)
        out.table = input
        out.mr = temp_mr
        out.stream = stream
        out.c_obj = move(obj)
        return out

    cpdef bool has_next(self):
        """
        Check if the packer has more chunks to pack.

        Returns
        -------
        True if the packer has chunks still to pack.
        """
        with nogil:
            return dereference(self.c_obj).has_next()

    cpdef size_t get_total_contiguous_size(self):
        """
        Get the total size of the packed data.

        Returns
        -------
        Size of packed data.
        """
        with nogil:
            return dereference(self.c_obj).get_total_contiguous_size()

    cpdef size_t next(self, DeviceBuffer buf):
        """
        Pack the next chunk into the provided device buffer.

        Parameters
        ----------
        buf
            The device buffer to use as a staging buffer, must be at
            least as large as the `user_buffer_size` used to construct the
            packer.

        Returns
        -------
        Number of bytes packed.

        Notes
        -----
        This is stream-ordered with respect to the stream used when
        creating the `ChunkedPack`.
        """
        cdef device_span[uint8_t] d_span = device_span[uint8_t](
            <uint8_t *>buf.c_data(), buf.c_size()
        )
        with nogil:
            return dereference(self.c_obj).next(d_span)

    cpdef memoryview build_metadata(self):
        """
        Build the metadata for the packed representation.

        Returns
        -------
        memoryview of metadata suitable for passing to `unpack_from_memoryviews`.
        """
        cdef unique_ptr[vector[uint8_t]] metadata
        with nogil:
            metadata = move(dereference(self.c_obj).build_metadata())
        return memoryview(HostBuffer.from_unique_ptr(move(metadata)))

    cpdef tuple pack_to_host(self, DeviceBuffer buf):
        """
        Pack the entire table into a host buffer.

        Parameters
        ----------
        buf
           The device buffer to use as a staging buffer, must be at
           least as large as the `user_buffer_size` used to construct the
           packer.

        Returns
        -------
        tuple of metadata and packed host data (as memoryviews)

        Notes
        -----
        This is stream-ordered with respect to the stream used when
        creating the `ChunkedPack` and syncs that stream before returning.

        Raises
        ------
        RuntimeError
            If the copy to host fails or an incorrectly sized buffer
            is provided.
        """
        cdef size_t offset = 0
        cdef size_t size
        cdef device_span[uint8_t] d_span = device_span[uint8_t](
            <uint8_t *>buf.c_data(), buf.c_size()
        )
        cdef cudaError_t err
        cdef unique_ptr[vector[uint8_t]] h_buf = (
            make_unique[vector[uint8_t]](
                dereference(self.c_obj).get_total_contiguous_size()
            )
        )
        cdef cuda_stream_view stream = self.stream.view()
        with nogil:
            while dereference(self.c_obj).has_next():
                size = dereference(self.c_obj).next(d_span)
                err = cudaMemcpyAsync(
                    dereference(h_buf).data() + offset,
                    d_span.data(),
                    size,
                    cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream.value(),
                )
                offset += size
                if err != cudaError.cudaSuccess:
                    stream.synchronize()
                    raise RuntimeError(
                        f"Memcpy in pack_to_host failed error: {err}"
                    )
        stream.synchronize()
        return (
            self.build_metadata(),
            memoryview(HostBuffer.from_unique_ptr(move(h_buf))),
        )


cpdef PackedColumns pack(Table input, Stream stream=None, DeviceMemoryResource mr=None):
    """Deep-copy a table into a serialized contiguous memory format.

    Later use `unpack` or `unpack_from_memoryviews` to unpack the serialized
    data back into the table.

    Parameters
    ----------
    input : Table
        Table to pack.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    PackedColumns
        The packed columns.

    Examples
    --------
    >>> packed = pylibcudf.contiguous_split.pack(...)
    >>> # Either unpack the whole `PackedColumns` at once.
    >>> pylibcudf.contiguous_split.unpack(packed)
    >>> # Or unpack the two serialized buffers in `PackedColumns`.
    >>> metadata, gpu_data = packed.release()
    >>> pylibcudf.contiguous_split.unpack_from_memoryviews(metadata, gpu_data)

    For details, see :cpp:func:`pack`.
    """
    cdef unique_ptr[packed_columns] pack
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    with nogil:
        pack = move(make_unique[packed_columns](
            cpp_pack(input.view(), stream.view(), mr.get_mr())
        ))
    return PackedColumns.from_libcudf(move(pack), stream, mr)


cpdef Table unpack(PackedColumns input, Stream stream=None):
    """Deserialize the result of `pack`.

    Copies the result of a serialized table into a table.

    For details, see :cpp:func:`unpack`.

    Parameters
    ----------
    input : PackedColumns
        The packed columns to unpack.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Table
        Copy of the packed columns.
    """
    cdef table_view v
    stream = _get_stream(stream)
    with nogil:
        v = cpp_unpack(dereference(input.c_obj))
    return Table.from_table_view_of_arbitrary(v, input, stream)


cpdef Table unpack_from_memoryviews(
    memoryview metadata,
    gpumemoryview gpu_data,
    Stream stream=None,
):
    """Deserialize the result of `pack`.

    Copies the result of a serialized table into a table.

    For details, see :cpp:func:`unpack`.

    Parameters
    ----------
    metadata : memoryview
        The packed metadata to unpack.
    gpu_data : gpumemoryview
        The packed gpu_data to unpack.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Table
        Copy of the packed columns.
    """
    stream = _get_stream(stream)
    if metadata.nbytes == 0:
        if gpu_data.__cuda_array_interface__["data"][0] != 0:
            raise ValueError("Expected an empty gpu_data from unpacking an empty table")
        # For an empty table we just attach the default mr since it will not be
        # used for any operations.
        return Table.from_libcudf(
            make_unique[table](table_view()),
            stream,
            _get_memory_resource(),
        )

    # Extract the raw data pointers
    cdef const uint8_t[::1] _metadata = metadata
    cdef const uint8_t* metadata_ptr = &_metadata[0]
    cdef const uint8_t* gpu_data_ptr = <uint8_t*>gpu_data.ptr

    cdef table_view v
    with nogil:
        v = cpp_unpack(metadata_ptr, gpu_data_ptr)
    return Table.from_table_view_of_arbitrary(v, gpu_data, stream)
