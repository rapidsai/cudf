# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from rmm._lib.device_uvector cimport device_uvector


ctypedef unique_ptr[device_uvector[size_type]] gather_map_type

cdef extern from "cudf/join.hpp" namespace "cudf" nogil:
    cdef pair[gather_map_type, gather_map_type] inner_join(
        const table_view left_keys,
        const table_view right_keys,
    ) except +

    cdef pair[gather_map_type, gather_map_type] left_join(
        const table_view left_keys,
        const table_view right_keys,
    ) except +

    cdef pair[gather_map_type, gather_map_type] full_join(
        const table_view left_keys,
        const table_view right_keys,
    ) except +

    cdef gather_map_type left_semi_join(
        const table_view left_keys,
        const table_view right_keys,
    ) except +

    cdef gather_map_type left_anti_join(
        const table_view left_keys,
        const table_view right_keys,
    ) except +
