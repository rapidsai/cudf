# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libc.stdint cimport int32_t


ctypedef bool underlying_type_t_order
ctypedef bool underlying_type_t_null_order
ctypedef bool underlying_type_t_sorted
ctypedef int32_t underlying_type_t_interpolation

cdef extern from "cudf/types.hpp" \
        namespace "cudf" nogil:

    ctypedef enum type_id:
        EMPTY "cudf::type_id::EMPTY"
        INT8 "cudf::type_id::INT8"
        INT16 "cudf::type_id::INT16"
        INT32 "cudf::type_id::INT32"
        INT64 "cudf::type_id::INT64"
        FLOAT32 "cudf::type_id::FLOAT32"
        FLOAT64 "cudf::type_id::FLOAT64"
        BOOL8 "cudf::type_id::BOOL8"
        TIMESTAMP_DAYS "cudf::type_id::TIMESTAMP_DAYS"
        TIMESTAMP_SECONDS "cudf::type_id::TIMESTAMP_SECONDS"
        TIMESTAMP_MILLISECONDS "cudf::type_id::TIMESTAMP_MILLISECONDS"
        TIMESTAMP_MICROSECONDS "cudf::type_id::TIMESTAMP_MICROSECONDS"
        TIMESTAMP_NANOSECONDS "cudf::type_id::TIMESTAMP_NANOSECONDS"
        DICTIONARY32 "cudf::type_id::DICTIONARY32"
        STRING "cudf::type_id::STRING"
        NUM_TYPE_IDS "cudf::type_id::NUM_TYPE_IDS"

    cdef cppclass data_type:
        type_id _id
        data_type(type_id id) except +
