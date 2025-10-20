# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.libcudf.utilities.span cimport host_span
from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "cudf/detail/utilities/stream_pool.hpp" namespace "cudf::detail" nogil:
    void join_streams(
        host_span[const cuda_stream_view] streams, cuda_stream_view stream
    ) except +
