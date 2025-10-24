# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.string cimport string
from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings.side_type cimport side_type
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column pad(
    Column input,
    size_type width,
    side_type side,
    str fill_char,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column zfill(
    Column input, size_type width, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column zfill_by_widths(
    Column input, Column widths, Stream stream=*, DeviceMemoryResource mr=*
)
