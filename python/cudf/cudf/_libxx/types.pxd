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
        EMPTY = 0,
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT32,
        FLOAT64,
        BOOL8,
        TIMESTAMP_DAYS,
        TIMESTAMP_SECONDS,
        TIMESTAMP_MILLISECONDS,
        TIMESTAMP_MICROSECONDS,
        TIMESTAMP_NAN,
        STRING,
        NUM_TYPE_IDS,

    cdef cppclass data_type:
        type_id _id
        data_type(type_id id) except +
