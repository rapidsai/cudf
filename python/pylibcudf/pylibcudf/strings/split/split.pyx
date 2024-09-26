# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
# from pylibcudf.libcudf.column.column cimport column
# from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_program cimport RegexProgram
# from pylibcudf.strings.split cimport split as cpp_split
from pylibcudf.table cimport Table


cpdef Table split(Column strings_column, Scalar delimiter, size_type maxsplit):
    pass

cpdef Table rsplit(Column strings_column, Scalar delimiter, size_type maxsplit):
    pass

cpdef Column split_record(Column strings, Scalar delimiter, size_type maxsplit):
    pass

cpdef Column rsplit_record(Column strings, Scalar delimiter, size_type maxsplit):
    pass

cpdef Table split_re(Column input, RegexProgram prog, size_type maxsplit):
    pass

cpdef Table rsplit_re(Column input, RegexProgram prog, size_type maxsplit):
    pass

cpdef Column split_record_re(Column input, RegexProgram prog, size_type maxsplit):
    pass

cpdef Column rsplit_record_re(Column input, RegexProgram prog, size_type maxsplit):
    pass
