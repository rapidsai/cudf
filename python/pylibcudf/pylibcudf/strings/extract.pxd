# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.table cimport Table


cpdef Table extract(Column input, RegexProgram prog)

cpdef Column extract_all_record(Column input, RegexProgram prog)
