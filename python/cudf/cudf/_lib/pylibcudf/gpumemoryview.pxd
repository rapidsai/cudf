# Copyright (c) 2023, NVIDIA CORPORATION.

from rmm._lib.device_buffer cimport DeviceBuffer


cdef class gpumemoryview:
    cdef readonly Py_ssize_t ptr
    cdef object base


cpdef gpumemoryview_from_device_buffer(DeviceBuffer buf)
