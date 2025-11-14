# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column replace(
    Column input,
    Scalar target,
    Scalar repl,
    size_type maxrepl=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
cpdef Column replace_multiple(
    Column input,
    Column target,
    Column repl,
    size_type maxrepl=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
cpdef Column replace_slice(
    Column input,
    Scalar repl=*,
    size_type start=*,
    size_type stop=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
