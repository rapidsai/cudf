# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.table cimport Table
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Table split(
    Column strings_column, Scalar delimiter, size_type maxsplit, Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Table rsplit(
    Column strings_column, Scalar delimiter, size_type maxsplit, Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column split_record(
    Column strings, Scalar delimiter, size_type maxsplit, Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column rsplit_record(
    Column strings, Scalar delimiter, size_type maxsplit, Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Table split_re(
    Column input, RegexProgram prog, size_type maxsplit, Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Table rsplit_re(
    Column input, RegexProgram prog, size_type maxsplit, Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column split_record_re(
    Column input, RegexProgram prog, size_type maxsplit, Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column rsplit_record_re(
    Column input, RegexProgram prog, size_type maxsplit, Stream stream=*,
    DeviceMemoryResource mr=*,
)
