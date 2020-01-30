from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cudf._libxx.lib cimport *

cdef extern from "cudf/join.hpp" namespace "cudf::experimental" nogil: 
    cdef unique_ptr[table] cpp_inner_join "cudf::experimental::inner_join" (
        table_view left,
        table_view right, 
        vector[int] left_on,
        vector[int] right_on,
        vector[pair[int, int]] columns_in_common
    )

    cdef unique_ptr[table] cpp_left_join "cudf::experimental::left_join" (
        table_view left,
        table_view right, 
        vector[int] left_on,
        vector[int] right_on,
        vector[pair[int, int]] columns_in_common
    )

    cdef unique_ptr[table] cpp_full_join "cudf::experimental::full_join" (
        table_view left,
        table_view right, 
        vector[int] left_on,
        vector[int] right_on,
        vector[pair[int, int]] columns_in_common
    )