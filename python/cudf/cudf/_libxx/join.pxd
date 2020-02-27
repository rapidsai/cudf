from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cudf._libxx.lib cimport *


cdef extern from "cudf/join.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] inner_join(
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on,
        const vector[pair[int, int]] columns_in_common
    ) except +
    cdef unique_ptr[table] left_join(
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on,
        const vector[pair[int, int]] columns_in_common
    ) except +
    cdef unique_ptr[table] full_join(
        const table_view left,
        const table_view right,
        const vector[int] left_on,
        const vector[int] right_on,
        const vector[pair[int, int]] columns_in_common
    ) except +
