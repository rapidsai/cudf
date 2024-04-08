# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

#from cudf._lib.cpp.strings cimport case as cpp_case
#from cudf._lib.cpp.column.column cimport column

#from .column cimport Column

#cpdef Column capitalize(Column input):
#    cdef unique_ptr[column] c_result
#    with nogil:
#        c_result = cpp_case.capitalize(input.view())
#    
#    return Column.from_libcudf(move(c_result))

