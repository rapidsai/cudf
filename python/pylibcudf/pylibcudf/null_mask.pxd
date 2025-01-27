# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.libcudf.types cimport mask_state, size_type

from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from .column cimport Column


cpdef DeviceBuffer copy_bitmask(Column col)

cpdef size_t bitmask_allocation_size_bytes(size_type number_of_bits)

cpdef DeviceBuffer create_null_mask(size_type size, mask_state state = *)

cpdef tuple bitmask_and(list columns)

cpdef tuple bitmask_or(list columns)

cpdef size_type null_count(Py_ssize_t bitmask, size_type start, size_type stop)
