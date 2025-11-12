# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column replace_tokens(
    Column input,
    Column targets,
    Column replacements,
    Scalar delimiter=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column filter_tokens(
    Column input,
    size_type min_token_length,
    Scalar replacement=*,
    Scalar delimiter=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
