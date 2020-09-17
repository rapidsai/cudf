# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr
from libc.stdint cimport uint8_t

from cudf._lib.cpp.types cimport data_type, size_type
cimport cudf._lib.cpp.io.types as cudf_io_types
cimport cudf._lib.cpp.table.table_view as cudf_table_view


cdef extern from "cudf/io/functions.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass read_orc_args:
        cudf_io_types.source_info source
        vector[string] columns
        vector[size_type] stripes
        size_type skip_rows
        size_type num_rows
        bool use_index
        bool use_np_dtypes
        data_type timestamp_type
        bool decimals_as_float
        int forced_decimals_scale

        read_orc_args() except +
        read_orc_args(cudf_io_types.source_info &src) except +

    cdef cudf_io_types.table_with_metadata read_orc(
        read_orc_args &args
    ) except +

    cdef cppclass write_orc_args:
        cudf_io_types.sink_info sink
        cudf_io_types.compression_type compression
        bool enable_statistics
        cudf_table_view.table_view table
        const cudf_io_types.table_metadata *metadata

        write_orc_args() except +
        write_orc_args(cudf_io_types.sink_info sink_,
                       cudf_table_view.table_view table_,
                       cudf_io_types.table_metadata *metadata_,
                       cudf_io_types.compression_type compression_,
                       bool enable_statistics_) except +

    cdef void write_orc(write_orc_args args) except +

    cdef unique_ptr[vector[uint8_t]] merge_rowgroup_metadata(
        const vector[unique_ptr[vector[uint8_t]]]& metadata_list
    ) except +
