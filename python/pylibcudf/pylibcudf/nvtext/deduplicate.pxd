# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

cpdef Column build_suffix_array(
    Column input,
    size_type min_width,
    Stream stream=*,
    DeviceMemoryResource mr=*
)
cpdef Column resolve_duplicates(
    Column input,
    Column indices,
    size_type min_width,
    Stream stream=*,
    DeviceMemoryResource mr=*
)
cpdef Column resolve_duplicates_pair(
    Column input1,
    Column indices1,
    Column input2,
    Column indices2,
    size_type min_width,
    Stream stream=*,
    DeviceMemoryResource mr=*
)
