# Copyright (c) 2024, NVIDIA CORPORATION.

from rmm._lib.device_buffer cimport DeviceBuffer

from cudf._lib.pylibcudf.libcudf.types cimport mask_state, size_type

from .column cimport Column


cpdef DeviceBuffer copy_bitmask(Column col)

cpdef size_t bitmask_allocation_size_bytes(size_type number_of_bits)

cpdef DeviceBuffer create_null_mask(size_type size, mask_state state = *)

cpdef tuple bitmask_and(list columns)

cpdef tuple bitmask_or(list columns)
