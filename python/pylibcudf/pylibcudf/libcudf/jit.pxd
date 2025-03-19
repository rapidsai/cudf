# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp cimport bool
from pylibcudf.exception_handler cimport libcudf_exception_handler

cdef extern from "cudf/jit/runtime_support.hpp" namespace "cudf" nogil:

    cdef bool is_runtime_jit_supported() except +libcudf_exception_handler
