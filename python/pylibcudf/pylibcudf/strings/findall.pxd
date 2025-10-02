# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.strings.regex_program cimport RegexProgram
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column find_re(
    Column input, RegexProgram pattern, Stream stream=*, DeviceMemoryResource mr=*
)
cpdef Column findall(
    Column input, RegexProgram pattern, Stream stream=*, DeviceMemoryResource mr=*
)
