# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

from pylibcudf.libcudf.types cimport size_type

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .types cimport DataType


cpdef Column interleave_columns(Table source_table)
cpdef Table tile(Table source_table, size_type count)
cpdef void table_to_array(
    Table input_table,
    uintptr_t ptr,
    size_t size,
    Stream stream=*
)
