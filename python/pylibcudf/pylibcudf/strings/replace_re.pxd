# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_flags cimport regex_flags
from pylibcudf.strings.regex_program cimport RegexProgram
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

ctypedef fused Replacement:
    Column
    Scalar

ctypedef fused Patterns:
    RegexProgram
    list


cpdef Column replace_re(
    Column input,
    Patterns patterns,
    Replacement replacement=*,
    size_type max_replace_count=*,
    regex_flags flags=*,
    Stream stream=*,
    DeviceMemoryResource mr=*
)

cpdef Column replace_with_backrefs(
    Column input,
    RegexProgram prog,
    str replacement,
    Stream stream=*,
    DeviceMemoryResource mr=*
)
