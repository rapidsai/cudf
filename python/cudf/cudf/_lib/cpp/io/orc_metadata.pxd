# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.vector cimport vector
from libcpp.string cimport string

cimport cudf._lib.cpp.io.types as cudf_io_types


cdef extern from "cudf/io/orc_metadata.hpp" \
        namespace "cudf::io" nogil:

    cdef vector[vector[string]] read_orc_statistics(
        cudf_io_types.source_info src_info
    ) except +
