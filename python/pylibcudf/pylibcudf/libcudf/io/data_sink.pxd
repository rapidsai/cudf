# Copyright (c) 2023, NVIDIA CORPORATION.


cdef extern from "cudf/io/data_sink.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass data_sink:
        pass
