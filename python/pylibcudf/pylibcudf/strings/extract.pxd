# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.table cimport Table
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Table extract(
    Column input, RegexProgram prog, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column extract_all_record(
    Column input, RegexProgram prog, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column extract_single(
    Column input,
    RegexProgram prog,
    size_type group,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
