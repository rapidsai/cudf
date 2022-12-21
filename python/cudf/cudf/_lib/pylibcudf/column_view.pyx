# Copyright (c) 2022, NVIDIA CORPORATION.

from enum import IntEnum

from libc.stdint cimport int32_t, uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport bitmask_type, data_type, size_type, type_id


# TODO: Unclear what the perf impacts of using pure Python enums here will be,
# but I think it will be necessary because ultimately we need these to be
# constructible from cudf (not just pylibcudf or custom Cython code using it).
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


# Helpers
# These should be extracted into separate modules.
cdef void * int_to_void_ptr(ptr):
    # Cython will not cast a Python integer directly to a pointer, so the
    # intermediate cast to a uintptr_t is necessary
    return <void*><uintptr_t>(ptr)


cdef bitmask_type * int_to_bitmask_ptr(ptr):
    # Cython will not cast a Python integer directly to a pointer, so the
    # intermediate cast to a uintptr_t is necessary
    return <bitmask_type*><uintptr_t>(ptr)


# Cython doesn't support scoped enumerations. It assumes that enums correspond
# to their underlying value types and will thus attempt operations that are
# invalid. This code will ensure that these values are explicitly cast to the
# underlying type before casting to the final type.
ctypedef int32_t underlying_type_t_type_id
cdef type_id py_type_to_c_type(py_type_id):
    return <type_id> (<underlying_type_t_type_id> py_type_id)


cdef class ColumnView:
    """Wrapper around column_view."""
    cdef unique_ptr[column_view] * c_obj

    # TODO: For now assuming data and mask are Buffers, but eventually need to
    # define a new gpumemoryview type to handle this. For that object it should
    # be possible to access all attributes via fast cdef functions (no Python
    # overhead for querying size etc).
    def __cinit__(self, py_type_id, object data_buf, object mask_buf):
        cdef type_id c_type_id = py_type_to_c_type(py_type_id)
        cdef data_type dtype = data_type(c_type_id)
        cdef size_type size = data_buf.size
        cdef const void * data = int_to_void_ptr(data_buf.ptr)
        cdef const bitmask_type * null_mask
        if mask_buf is not None:
            null_mask = int_to_bitmask_ptr(mask_buf.ptr)
        # TODO: At the moment libcudf does not expose APIs for counting the
        # nulls in a bitmask directly (those APIs are in detail/null_mask). If
        # we want to allow more flexibility in the Cython layer we'll need to
        # expose those eventually. This dovetails with our desire to expose
        # other functionality too like bitmask_and.
        cdef size_type null_count = 0
        # TODO: offset and children not yet supported
        cdef size_type offset = 0
        cdef const vector[column_view] children

        self.c_obj.reset(
            new column_view(
                dtype, size, data, null_mask, null_count, offset, children
            )
        )
