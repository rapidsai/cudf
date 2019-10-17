# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *


cdef extern from "cudf/cudf.h" nogil:

    ctypedef enum gdf_input_type:
        FILE_PATH = 0,
        HOST_BUFFER,
