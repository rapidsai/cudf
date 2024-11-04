# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport null_equality, size_type

from rmm.librmm.device_uvector cimport device_uvector

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

    cdef gather_map_pair_type conditional_inner_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type conditional_inner_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        optional[size_t] output_size
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type conditional_left_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type conditional_left_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        optional[size_t] output_size
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type conditional_full_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type conditional_full_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        optional[size_t] output_size
    ) except +libcudf_exception_handler

    cdef gather_map_type conditional_left_semi_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
    ) except +libcudf_exception_handler

    cdef gather_map_type conditional_left_semi_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        optional[size_t] output_size
    ) except +libcudf_exception_handler

    cdef gather_map_type conditional_left_anti_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
    ) except +libcudf_exception_handler

    cdef gather_map_type conditional_left_anti_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        optional[size_t] output_size
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type mixed_inner_join(
        const table_view left_equality,
        const table_view right_equality,
        const table_view left_conditional,
        const table_view right_conditional,
        const expression binary_predicate,
        null_equality compare_nulls
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type mixed_left_join(
        const table_view left_equality,
        const table_view right_equality,
        const table_view left_conditional,
        const table_view right_conditional,
        const expression binary_predicate,
        null_equality compare_nulls
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type mixed_full_join(
        const table_view left_equality,
        const table_view right_equality,
        const table_view left_conditional,
        const table_view right_conditional,
        const expression binary_predicate,
        null_equality compare_nulls
    ) except +libcudf_exception_handler

    cdef gather_map_type mixed_left_semi_join(
        const table_view left_equality,
        const table_view right_equality,
        const table_view left_conditional,
        const table_view right_conditional,
        const expression binary_predicate,
        null_equality compare_nulls
    ) except +libcudf_exception_handler

    cdef gather_map_type mixed_left_anti_join(
        const table_view left_equality,
        const table_view right_equality,
        const table_view left_conditional,
        const table_view right_conditional,
        const expression binary_predicate,
        null_equality compare_nulls
    ) except +libcudf_exception_handler
