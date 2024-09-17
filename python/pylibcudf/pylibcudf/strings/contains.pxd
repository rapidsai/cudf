# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_program cimport RegexProgram


cpdef Column contains_re(Column input, RegexProgram prog)

cpdef Column count_re(Column input, RegexProgram prog)

cpdef Column matches_re(Column input, RegexProgram prog)

cpdef Column like(
    Column input,
    Column pattern,
    Scalar escape_character = *
)
