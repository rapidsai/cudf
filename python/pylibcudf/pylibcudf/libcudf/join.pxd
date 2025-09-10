# Copyright (c) 2020-2025, NVIDIA CORPORATION.

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
from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from rmm.librmm.device_uvector cimport device_uvector
from pylibcudf.libcudf.utilities.span cimport device_span

ctypedef unique_ptr[device_uvector[size_type]] gather_map_type
ctypedef pair[gather_map_type, gather_map_type] gather_map_pair_type
ctypedef optional[pair[size_t, device_span[const size_type]]] output_size_data_type

cdef extern from "cudf/join/join.hpp" namespace "cudf" nogil:
    cdef gather_map_pair_type inner_join(
        const table_view left_keys,
        const table_view right_keys,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type left_join(
        const table_view left_keys,
        const table_view right_keys,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type full_join(
        const table_view left_keys,
        const table_view right_keys,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_type left_semi_join(
        const table_view left_keys,
        const table_view right_keys,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_type left_anti_join(
        const table_view left_keys,
        const table_view right_keys,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type inner_join(
        const table_view left_keys,
        const table_view right_keys,
        null_equality nulls_equal,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type left_join(
        const table_view left_keys,
        const table_view right_keys,
        null_equality nulls_equal,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type full_join(
        const table_view left_keys,
        const table_view right_keys,
        null_equality nulls_equal,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_type left_semi_join(
        const table_view left_keys,
        const table_view right_keys,
        null_equality nulls_equal,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_type left_anti_join(
        const table_view left_keys,
        const table_view right_keys,
        null_equality nulls_equal,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] cross_join(
        const table_view left,
        const table_view right,
        cuda_stream_view stream
    ) except +

cdef extern from "cudf/join/conditional_join.hpp" namespace "cudf" nogil:
    cdef gather_map_pair_type conditional_inner_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type conditional_inner_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        optional[size_t] output_size,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type conditional_left_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type conditional_left_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        optional[size_t] output_size,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type conditional_full_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_type conditional_left_semi_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_type conditional_left_semi_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        optional[size_t] output_size,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_type conditional_left_anti_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_type conditional_left_anti_join(
        const table_view left,
        const table_view right,
        const expression binary_predicate,
        optional[size_t] output_size,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

cdef extern from "cudf/join/mixed_join.hpp" namespace "cudf" nogil:
    cdef gather_map_pair_type mixed_inner_join(
        const table_view left_equality,
        const table_view right_equality,
        const table_view left_conditional,
        const table_view right_conditional,
        const expression binary_predicate,
        null_equality compare_nulls,
        output_size_data_type output_size_data,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type mixed_left_join(
        const table_view left_equality,
        const table_view right_equality,
        const table_view left_conditional,
        const table_view right_conditional,
        const expression binary_predicate,
        null_equality compare_nulls,
        output_size_data_type output_size_data,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_pair_type mixed_full_join(
        const table_view left_equality,
        const table_view right_equality,
        const table_view left_conditional,
        const table_view right_conditional,
        const expression binary_predicate,
        null_equality compare_nulls,
        output_size_data_type output_size_data,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_type mixed_left_semi_join(
        const table_view left_equality,
        const table_view right_equality,
        const table_view left_conditional,
        const table_view right_conditional,
        const expression binary_predicate,
        null_equality compare_nulls,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef gather_map_type mixed_left_anti_join(
        const table_view left_equality,
        const table_view right_equality,
        const table_view left_conditional,
        const table_view right_conditional,
        const expression binary_predicate,
        null_equality compare_nulls,
        cuda_stream_view stream
    ) except +libcudf_exception_handler
