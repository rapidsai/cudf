# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.strings.regex_program cimport RegexProgram


cpdef Column find_re(Column input, RegexProgram pattern)
cpdef Column findall(Column input, RegexProgram pattern)
