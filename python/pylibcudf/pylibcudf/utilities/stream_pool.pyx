# Copyright (c) 2025, NVIDIA CORPORATION.
from libcpp.vector cimport vector
from pylibcudf.libcudf.utilities cimport stream_pool as cpp_stream_pool
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

ctypedef const cuda_stream_view const_cuda_stream_view

__all__ = ["join_streams"]


cpdef void join_streams(list streams, Stream stream):
    """Synchronize a stream to an event on a set of streams.

    After joining the streams, data that is valid on any of the
    streams in ``streams`` is also valid on ``stream``.

    Parameters
    ----------
    streams
        Streams to wait on.
    stream
        Joined stream that synchronizes with the waited-on streams.
    """
    cdef vector[cuda_stream_view] c_streams = [
        (<Stream?>stream).view() for stream in streams
    ]
    cdef Stream c_stream = <Stream?>stream
    with nogil:
        cpp_stream_pool.join_streams(c_streams, c_stream.view())
