# Copyright (c) 2023, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp cimport bool as cbool

from cudf._lib.cpp.types cimport data_type, interpolation, null_policy, type_id

ctypedef int32_t underlying_type_t_type_id


# Enum representing possible data type ids. This is the Cython representation
# of libcudf's type_id.
cpdef enum TypeId:
    EMPTY = <underlying_type_t_type_id> type_id.EMPTY
    INT8 = <underlying_type_t_type_id> type_id.INT8
    INT16 = <underlying_type_t_type_id> type_id.INT16
    INT32 = <underlying_type_t_type_id> type_id.INT32
    INT64 = <underlying_type_t_type_id> type_id.INT64
    UINT8 = <underlying_type_t_type_id> type_id.UINT8
    UINT16 = <underlying_type_t_type_id> type_id.UINT16
    UINT32 = <underlying_type_t_type_id> type_id.UINT32
    UINT64 = <underlying_type_t_type_id> type_id.UINT64
    FLOAT32 = <underlying_type_t_type_id> type_id.FLOAT32
    FLOAT64 = <underlying_type_t_type_id> type_id.FLOAT64
    BOOL8 = <underlying_type_t_type_id> type_id.BOOL8
    TIMESTAMP_DAYS = <underlying_type_t_type_id> type_id.TIMESTAMP_DAYS
    TIMESTAMP_SECONDS = <underlying_type_t_type_id> type_id.TIMESTAMP_SECONDS
    TIMESTAMP_MILLISECONDS = (
        <underlying_type_t_type_id> type_id.TIMESTAMP_MILLISECONDS
    )
    TIMESTAMP_MICROSECONDS = (
        <underlying_type_t_type_id> type_id.TIMESTAMP_MICROSECONDS
    )
    TIMESTAMP_NANOSECONDS = (
        <underlying_type_t_type_id> type_id.TIMESTAMP_NANOSECONDS
    )
    DICTIONARY32 = <underlying_type_t_type_id> type_id.DICTIONARY32
    STRING = <underlying_type_t_type_id> type_id.STRING
    LIST = <underlying_type_t_type_id> type_id.LIST
    STRUCT = <underlying_type_t_type_id> type_id.STRUCT
    NUM_TYPE_IDS = <underlying_type_t_type_id> type_id.NUM_TYPE_IDS
    DURATION_SECONDS = <underlying_type_t_type_id> type_id.DURATION_SECONDS
    DURATION_MILLISECONDS = (
        <underlying_type_t_type_id> type_id.DURATION_MILLISECONDS
    )
    DURATION_MICROSECONDS = (
        <underlying_type_t_type_id> type_id.DURATION_MICROSECONDS
    )
    DURATION_NANOSECONDS = (
        <underlying_type_t_type_id> type_id.DURATION_NANOSECONDS
    )
    DECIMAL32 = <underlying_type_t_type_id> type_id.DECIMAL32
    DECIMAL64 = <underlying_type_t_type_id> type_id.DECIMAL64
    DECIMAL128 = <underlying_type_t_type_id> type_id.DECIMAL128


cdef type_id py_type_to_c_type(TypeId py_type_id) nogil


cdef class DataType:
    cdef data_type c_obj

    cpdef TypeId id(self)
    cpdef int32_t scale(self)

    @staticmethod
    cdef DataType from_libcudf(data_type dt)
