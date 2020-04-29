# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.io.types cimport source_info

cdef source_info make_source_info(src) except*
