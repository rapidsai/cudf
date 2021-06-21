# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.io.types cimport data_sink, sink_info, source_info


cdef source_info make_source_info(list src) except*
cdef sink_info make_sink_info(src, unique_ptr[data_sink] & data) except*
