# Copyright (c) 2023, NVIDIA CORPORATION.


cdef extern from "exception_handler.hpp" namespace "cudf_python::exceptions":
    cdef void cudf_exception_handler()
