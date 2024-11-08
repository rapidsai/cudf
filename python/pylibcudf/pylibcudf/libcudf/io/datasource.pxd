# Copyright (c) 2023, NVIDIA CORPORATION.


cdef extern from "cudf/io/datasource.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass datasource:
        pass
