# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.strings.regex_program cimport RegexProgram


cpdef Column contains_re(Column input, RegexProgram prog)
