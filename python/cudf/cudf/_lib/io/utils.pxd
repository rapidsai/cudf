# Copyright (c) 2020, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.io.types cimport source_info, sink_info

cdef source_info make_source_info(src) except*
cdef unique_ptr[sink_info] make_sink_info(src) except*
