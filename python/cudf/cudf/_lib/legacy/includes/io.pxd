# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.legacy.cudf cimport *


cdef extern from "cudf/cudf.h" nogil:

    ctypedef enum gdf_input_type:
        FILE_PATH = 0,
        HOST_BUFFER,
