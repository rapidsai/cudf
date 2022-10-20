# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.vector cimport vector

cimport cudf._lib.cpp.io.types as cudf_io_types


cdef extern from "cudf/io/orc_metadata.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass raw_orc_statistics:
        vector[string] column_names
        vector[string] file_stats
        vector[vector[string]] stripes_stats

    cdef raw_orc_statistics read_raw_orc_statistics(
        cudf_io_types.source_info src_info
    ) except +
