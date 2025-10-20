# Copyright (c) 2025, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibcudf.libcudf.utilities cimport default_stream
from pylibcudf.libcudf.utilities cimport stream_pool as cpp_stream_pool
from pylibcudf.libcudf.utilities.span cimport host_span
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

ctypedef const cuda_stream_view const_cuda_stream_view

__all__ = ["is_ptds_enabled", "join_streams"]


cpdef bool is_ptds_enabled():
    """Checks if per-thread default stream is enabled.

    For details, see :cpp:func:`is_ptds_enabled`.
    """
    return default_stream.is_ptds_enabled()


cpdef void join_streams(list streams, Stream stream):
    """Synchronize a stream to an event on a set of streams.

    For details, see :cpp:func:`join_streams`.

    Parameters
    ----------
    streams : list[Stream]
        Streams to wait on.
    stream : Stream
        Joined stream that synchronizes with the waited-on streams.
    """
    if stream is None:
        raise TypeError(
            f"stream must be a Stream, got {type(stream).__name__} instead."
        )

    cdef vector[cuda_stream_view] c_streams
    cdef Stream s
    cdef Stream main_stream = <Stream>stream
    c_streams.reserve(len(streams))
    for item in streams:
        if not isinstance(item, Stream):
            raise TypeError(f"Expected Stream, got {type(item).__name__}")
        s = <Stream>item
        c_streams.push_back(s.view())

    cdef host_span[const_cuda_stream_view] streams_span = (
        host_span[const_cuda_stream_view](
            c_streams.data(), c_streams.size()
        )
    )

    with nogil:
        cpp_stream_pool.join_streams(streams_span, main_stream.view())
