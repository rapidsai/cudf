# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport null_equality, size_type

from rmm._lib.device_uvector cimport device_uvector

ctypedef unique_ptr[device_uvector[size_type]] gather_map_type
ctypedef pair[gather_map_type, gather_map_type] gather_map_pair_type

cdef extern from "cudf/join.hpp" namespace "cudf" nogil:
    cdef gather_map_pair_type inner_join(
        const table_view left_keys,
        const table_view right_keys,
    ) except +

    cdef gather_map_pair_type left_join(
        const table_view left_keys,
        const table_view right_keys,
    ) except +

    cdef gather_map_pair_type full_join(
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

    cdef gather_map_pair_type inner_join(
        const table_view left_keys,
        const table_view right_keys,
        null_equality nulls_equal,
    ) except +

    cdef gather_map_pair_type left_join(
        const table_view left_keys,
        const table_view right_keys,
        null_equality nulls_equal,
    ) except +

    cdef gather_map_pair_type full_join(
        const table_view left_keys,
        const table_view right_keys,
        null_equality nulls_equal,
    ) except +

    cdef gather_map_type left_semi_join(
        const table_view left_keys,
        const table_view right_keys,
        null_equality nulls_equal,
    ) except +

    cdef gather_map_type left_anti_join(
        const table_view left_keys,
        const table_view right_keys,
        null_equality nulls_equal,
    ) except +

    cdef unique_ptr[table] cross_join(
        const table_view left,
        const table_view right,
    ) except +
