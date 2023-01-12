# Copyright (c) 2023, NVIDIA CORPORATION.

from enum import IntEnum

from libc.stdint cimport int32_t

from cudf._lib.cpp.types cimport type_id


# TODO: Python enums are fairly slow. We can't use Cython enums since we need
# something usable from Python. One option is to remove the inheritance from
# IntEnum and adding a new staticmethod for constructing from a value rather
# than the call operator of the Enum metaclass. That won't work with a cdef
# class though.
# CONSIDER USING A cpdef enum. NEED TO FIGURE OUT how the APIs will look for
# that though (do cdef APIs accept both?)
class TypeId(IntEnum):
    EMPTY = type_id.EMPTY
    INT8 = type_id.INT8
    INT16 = type_id.INT16
    INT32 = type_id.INT32
    INT64 = type_id.INT64
    UINT8 = type_id.UINT8
    UINT16 = type_id.UINT16
    UINT32 = type_id.UINT32
    UINT64 = type_id.UINT64
    FLOAT32 = type_id.FLOAT32
    FLOAT64 = type_id.FLOAT64
    BOOL8 = type_id.BOOL8
    TIMESTAMP_DAYS = type_id.TIMESTAMP_DAYS
    TIMESTAMP_SECONDS = type_id.TIMESTAMP_SECONDS
    TIMESTAMP_MILLISECONDS = type_id.TIMESTAMP_MILLISECONDS
    TIMESTAMP_MICROSECONDS = type_id.TIMESTAMP_MICROSECONDS
    TIMESTAMP_NANOSECONDS = type_id.TIMESTAMP_NANOSECONDS
    DICTIONARY32 = type_id.DICTIONARY32
    STRING = type_id.STRING
    LIST = type_id.LIST
    STRUCT = type_id.STRUCT
    NUM_TYPE_IDS = type_id.NUM_TYPE_IDS
    DURATION_SECONDS = type_id.DURATION_SECONDS
    DURATION_MILLISECONDS = type_id.DURATION_MILLISECONDS
    DURATION_MICROSECONDS = type_id.DURATION_MICROSECONDS
    DURATION_NANOSECONDS = type_id.DURATION_NANOSECONDS
    DECIMAL32 = type_id.DECIMAL32
    DECIMAL64 = type_id.DECIMAL64
    DECIMAL128 = type_id.DECIMAL128


# Cython doesn't support scoped enumerations. It assumes that enums correspond
# to their underlying value types and will thus attempt operations that are
# invalid. This code will ensure that these values are explicitly cast to the
# underlying type before casting to the final type.
ctypedef int32_t underlying_type_t_type_id
cdef type_id py_type_to_c_type(py_type_id):
    return <type_id> (<underlying_type_t_type_id> py_type_id)


cdef class DataType:
    def __cinit__(self, id, int32_t scale=0):
        self.c_obj = data_type(py_type_to_c_type(id), scale)

    # TODO: Consider making both id and scale cached properties.
    cpdef id(self):
        return TypeId(self.c_obj.id())

    cpdef int32_t scale(self):
        return self.c_obj.scale()
