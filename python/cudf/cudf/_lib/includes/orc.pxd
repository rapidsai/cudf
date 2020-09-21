# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "cudf/cudf.h" namespace "cudf::io::orc" nogil:

    cdef cppclass reader_options:
        vector[string] columns
        bool use_index
        bool use_np_dtypes
        gdf_time_unit timestamp_unit
        bool decimals_as_float
        int forced_decimals_scale

        reader_options() except +

        reader_options(
            vector[string] columns,
            bool use_index,
            bool use_np_dtypes,
            gdf_time_unit timestamp_unit,
            bool decimals_as_float,
            int forced_decimals_scale
        ) except +

    cdef cppclass reader:
        reader(
            string filepath,
            const reader_options &args
        ) except +

        reader(
            const char *buffer,
            size_t length,
            const reader_options &args
        ) except +

        cudf_table read_all() except +

        cudf_table read_rows(size_t skip_rows, size_t num_rows) except +

        cudf_table read_stripe(size_t stripe) except +

    cdef enum compression_type:
        none 'cudf::io::orc::compression_type::none'
        snappy 'cudf::io::orc::compression_type::snappy'

    cdef cppclass writer_options:
        compression_type compression

        writer_options() except +

        writer_options(compression_type comp) except +

    cdef cppclass writer:
        writer(
            string filepath,
            const writer_options &args
        ) except +

        void write_all(const cudf_table &table) except +
