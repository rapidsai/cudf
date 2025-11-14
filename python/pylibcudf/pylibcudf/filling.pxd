# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table

ctypedef fused ColumnOrSize:
    Column
    size_type

cpdef Column fill(
    Column destination,
    size_type begin,
    size_type end,
    Scalar value,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef void fill_in_place(
    Column destination,
    size_type c_begin,
    size_type c_end,
    Scalar value,
    Stream stream = *,
)

cpdef Column sequence(
    size_type size,
    Scalar init,
    Scalar step,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table repeat(
    Table input_table,
    ColumnOrSize count,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column calendrical_month_sequence(
    size_type n,
    Scalar init,
    size_type months,
    Stream stream = *,
    DeviceMemoryResource mr = *,
)
