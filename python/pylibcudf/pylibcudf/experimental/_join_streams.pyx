# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.vector cimport vector

from pylibcudf.libcudf.detail.utilities cimport stream_pool as cpp_stream_pool
from pylibcudf.libcudf.utilities.span cimport host_span

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

ctypedef const cuda_stream_view const_cuda_stream_view


__all__ = ["join_streams"]


cpdef void join_streams(list streams, Stream stream):
    """Synchronize a stream to an event on a set of streams.

    This function synchronizes the joined stream with the waited-on streams
    by placing events on each of the waited-on streams and having the joined
    stream wait on those events.

    Parameters
    ----------
    streams : list
        A list of Stream objects to wait on.
    stream : Stream
        The joined stream that synchronizes with the waited-on streams.

    Examples
    --------
    >>> import pylibcudf as plc
    >>> from rmm.pylibrmm.stream import Stream
    >>> # Create streams
    >>> stream1 = Stream()
    >>> stream2 = Stream()
    >>> join_stream = Stream()
    >>> # ... do work on stream1 and stream2 ...
    >>> # Wait for both streams before continuing work on join_stream
    >>> plc.experimental.join_streams([stream1, stream2], join_stream)
    >>> # ... continue work on join_stream ...
    """
    cdef Stream c_stream = <Stream?>stream
    cdef vector[cuda_stream_view] c_streams

    c_streams.reserve(len(streams))
    for s in streams:
        c_streams.push_back((<Stream?>s).view())

    with nogil:
        cpp_stream_pool.join_streams(
            host_span[const_cuda_stream_view](c_streams.data(), c_streams.size()),
            c_stream.view()
        )
