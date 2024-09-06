# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.strings.regex_program cimport RegexProgram


cpdef Column contains_re(Column input, RegexProgram prog)
