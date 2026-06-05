# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libcpp.memory cimport make_unique, shared_ptr, unique_ptr
from libcpp.utility cimport move
from pylibcudf.expressions cimport Expression
from pylibcudf.io.parquet cimport ParquetReaderOptions
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.io.parquet cimport parquet_reader_options
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.streaming.chunks.arbitrary cimport cpp_OwningWrapper
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.actor cimport CppActor, cpp_Actor
from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context


cdef extern from "<cudf_streaming/streaming/parquet.hpp>" nogil:
    cdef cppclass cpp_Filter "cudf_streaming::streaming::Filter":
        cpp_Filter(cuda_stream_view, expression, cpp_OwningWrapper)

    cdef cpp_Actor cpp_read_parquet \
        "cudf_streaming::streaming::actor::read_parquet"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Communicator] comm,
            shared_ptr[cpp_Channel] ch_out,
            size_t num_producers,
            parquet_reader_options options,
            size_type num_rows_per_chunk,
            unique_ptr[cpp_Filter],
        ) except +ex_handler


cdef class Filter:
    """
    A filter expression for parquet reads.

    Parameters
    ----------
    stream
        The stream any scalars in the expression are valid on.
    expression
        The filter expression

    Notes
    -----
    The object safely manages the lifetime of the expressions when called
    from C++ coroutines, so it is safe to drop the expression passed in on
    the python side.
    """
    cdef unique_ptr[cpp_Filter] _handle

    def __init__(self, Stream stream not None, Expression filter not None):
        Py_INCREF(filter)
        self._handle = make_unique[cpp_Filter](
            stream.view(),
            deref(filter.c_obj),
            cpp_OwningWrapper(
                <void *><PyObject *>filter, py_deleter
            )
        )

    cdef unique_ptr[cpp_Filter] release_handle(self):
        """
        Move the owning C++ handle out of the object.

        Returns
        -------
        unique_ptr to the C++ Filter object.

        Raises
        ------
        ValueError
            If this Filter has already been used and the handle is already released.
        """
        if not self._handle:
            raise ValueError("Filter is uninitialized, has it been released?")
        return move(self._handle)

    def __dealloc__(self):
        with nogil:
            self._handle.reset()


def read_parquet(
    Context ctx not None,
    Communicator comm not None,
    Channel ch_out not None,
    size_t num_producers,
    ParquetReaderOptions options not None,
    size_type num_rows_per_chunk,
    Filter filter = None,
):
    """
    Create a streaming actor to read from parquet.

    Parameters
    ----------
    ctx
        Streaming execution context.
    comm
        The communicator.
    ch_out
        Output channel to receive the TableChunks.
    num_producers
        Number of concurrent producers of output chunks.
    options
        Reader options.
    num_rows_per_chunk
        Target (maximum) number of rows per output chunk.
    filter
        Optional filter object. If provided, is consumed by this function
        and not subsequently usable.

    Notes
    -----
    This is a collective operation, all ranks participating via the
    communicator must call it with the same options.
    """
    cdef cpp_Actor _ret
    cdef unique_ptr[cpp_Filter] c_filter
    if filter is not None:
        c_filter = move(filter.release_handle())
    with nogil:
        _ret = cpp_read_parquet(
            ctx._handle,
            comm._handle,
            ch_out._handle,
            num_producers,
            options.c_obj,
            num_rows_per_chunk,
            move(c_filter)
        )
    return CppActor.from_handle(
        make_unique[cpp_Actor](move(_ret)), owner=None
    )
