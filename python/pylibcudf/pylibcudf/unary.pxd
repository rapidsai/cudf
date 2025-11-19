# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from pylibcudf.libcudf.unary cimport unary_operator
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .types cimport DataType


cpdef Column unary_operation(
    Column input, unary_operator op, Stream stream = *, DeviceMemoryResource mr = *
)

cpdef Column is_null(Column input, Stream stream = *, DeviceMemoryResource mr = *)

cpdef Column is_valid(Column input, Stream stream = *, DeviceMemoryResource mr = *)

cpdef Column cast(
    Column input, DataType data_type, Stream stream = *, DeviceMemoryResource mr = *
)

cpdef Column is_nan(Column input, Stream stream = *, DeviceMemoryResource mr = *)

cpdef Column is_not_nan(Column input, Stream stream = *, DeviceMemoryResource mr = *)

cpdef bool is_supported_cast(DataType from_, DataType to)
