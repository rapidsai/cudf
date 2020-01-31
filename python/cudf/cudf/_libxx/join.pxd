from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cudf._libxx.lib cimport *

cdef extern from "cudf/join.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] cpp_inner_join "cudf::experimental::inner_join" (
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on,
        const vector[pair[int, int]] columns_in_common
    ) except +
cdef extern from "cudf/join.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] cpp_left_join "cudf::experimental::left_join" (
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on,
        const vector[pair[int, int]] columns_in_common
    ) except +
cdef extern from "cudf/join.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] cpp_full_join "cudf::experimental::full_join" (
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on,
        const vector[pair[int, int]] columns_in_common
    ) except +