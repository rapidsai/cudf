# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport make_unique, unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer

from cudf._lib.pylibcudf.libcudf cimport null_mask as cpp_null_mask
from cudf._lib.pylibcudf.libcudf.types cimport mask_state, size_type

from cudf._lib.pylibcudf.libcudf.types import \
    mask_state as MaskState  # no-cython-lint

from .column cimport Column
from .table cimport Table


cpdef DeviceBuffer copy_bitmask(Column col):
    cdef device_buffer db
    cdef unique_ptr[device_buffer] up_db

    with nogil:
        db = move(cpp_null_mask.copy_bitmask(col.view()))
        up_db = move(make_unique[device_buffer](move(db)))

    return DeviceBuffer.c_from_unique_ptr(move(up_db))


cpdef size_t bitmask_allocation_size_bytes(size_type number_of_bits):
    cdef size_t output_size

    with nogil:
        output_size = cpp_null_mask.bitmask_allocation_size_bytes(number_of_bits)

    return output_size


cpdef DeviceBuffer create_null_mask(
    size_type size,
    mask_state state = mask_state.UNINITIALIZED
):
    cdef device_buffer db
    cdef unique_ptr[device_buffer] up_db

    with nogil:
        db = move(cpp_null_mask.create_null_mask(size, state))
        up_db = move(make_unique[device_buffer](move(db)))

    return DeviceBuffer.c_from_unique_ptr(move(up_db))


cpdef tuple bitmask_and(list columns):
    cdef Table c_table = Table(columns)
    cdef pair[device_buffer, size_type] c_result
    cdef unique_ptr[device_buffer] up_db

    with nogil:
        c_result = move(cpp_null_mask.bitmask_and(c_table.view()))
        up_db = move(make_unique[device_buffer](move(c_result.first)))

    dbuf = DeviceBuffer.c_from_unique_ptr(move(up_db))

    return dbuf, c_result.second


cpdef tuple bitmask_or(list columns):
    cdef Table c_table = Table(columns)
    cdef pair[device_buffer, size_type] c_result
    cdef unique_ptr[device_buffer] up_db

    with nogil:
        c_result = move(cpp_null_mask.bitmask_or(c_table.view()))
        up_db = move(make_unique[device_buffer](move(c_result.first)))

    dbuf = DeviceBuffer.c_from_unique_ptr(move(up_db))

    return dbuf, c_result.second
