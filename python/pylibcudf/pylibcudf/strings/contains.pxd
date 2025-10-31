# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.strings.regex_program cimport RegexProgram
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column contains_re(
    Column input, RegexProgram prog, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column count_re(
    Column input, RegexProgram prog, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column matches_re(
    Column input, RegexProgram prog, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column like(
    Column input,
    str pattern,
    str escape_character=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
