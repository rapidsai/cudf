from libc.stdint cimport int32_t, uint32_t
from libcpp cimport bool
from libcpp.vector cimport vector

from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer

cdef extern from "<utility>" namespace "std" nogil:
    cdef device_buffer move(device_buffer)

cdef extern from "types.hpp" namespace "cudf" nogil:
    ctypedef int32_t size_type
    ctypedef uint32_t bitmask_type

    cdef enum:
        UNKOWN_NULL_COUNT = -1

    cdef enum type_id:
        EMPTY = 0
        INT8 = 1
        INT16 = 2
        INT32 = 3
        INT64 = 4
        FLOAT32 = 5
        FLOAT64 = 6
        BOOL8 = 7
        TIMESTAMP_DAYS = 8
        TIMESTAMP_SECONDS = 9
        TIMESTAMP_MILLISECONDS = 10
        TIMESTAMP_MICROSECONDS = 11
        TIMESTAMP_NANOSECONDS = 12
        CATEGORY = 13
        STRING = 14
        NUM_TYPE_IDS = 15

    cdef cppclass data_type:
        data_type()
        data_type(const data_type&)
        data_type(type_id id)
        type_id id()

cdef extern from "column/column.hpp" namespace "cudf" nogil:
    cdef cppclass column:
        column()
        column(const column& other)
        column(data_type dtype, size_type size, device_buffer&& data)
        size_type size()
        column_view view()
        mutable_column_view mutable_view()

cdef extern from "column/column_view.hpp" namespace "cudf" nogil:
    cdef cppclass column_view:
        column_view()
        column_view(const column_view& other)

        column_view& operator=(const column_view&)
        column_view& operator=(column_view&&)
        column_view(data_type type, size_type size, const void* data,
                    const bitmask_type* null_mask=NULL,
                    size_type null_count=UNKNOWN_NULL_COUNT,
                    size_type offset=0, const vector[column_view]& children=[])

        T* data[T]()
        bitmask_type* null_mask()
        size_type size()
        data_type type()
        bool nullable()
        size_type null_count()
        bool has_nulls()
        const bitmask_type * null_mask()
        size_type offset()

    cdef cppclass mutable_column_view:
        mutable_column_view()
        mutable_column_view(const mutable_column_view&)
        mutable_column_view& operator=(const mutable_column_view&)
        mutablecolumn_view(data_type type, size_type size, const void* data,
                           const bitmask_type* null_mask=NULL,
                           size_type null_count=UNKNOWN_NULL_COUNT,
                           size_type offset=0,const vector[mutable_column_view]& children=[])
        T* data[T]()
        bitmask_type* null_mask()
        size_type size()
        data_type type()
        bool nullable()
        size_type null_count()
        bool has_nulls()
        const bitmask_type * null_mask()
        size_type offset()
