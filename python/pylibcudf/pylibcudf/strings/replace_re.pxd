# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_flags cimport regex_flags
from pylibcudf.strings.regex_program cimport RegexProgram


cpdef Column replace_re(
    Column input,
    RegexProgram pattern,
    Scalar replacement=*,
    size_type max_replace_count=*,
)

cpdef Column replace_re_multi(
    Column input,
    list patterns,
    Column replacements,
    regex_flags flags=*,
)

cpdef Column replace_with_backrefs(
    Column input,
    RegexProgram prog,
    str replacement
)
