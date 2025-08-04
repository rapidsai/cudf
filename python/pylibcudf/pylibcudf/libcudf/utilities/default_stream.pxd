# Copyright (c) 2025, NVIDIA CORPORATION.
from libcpp cimport bool


cdef extern from "cudf/utilities/default_stream.hpp" namespace "cudf" nogil:
    cdef bool is_ptds_enabled()
