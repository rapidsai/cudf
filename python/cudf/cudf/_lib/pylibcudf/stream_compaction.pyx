# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp cimport stream_compaction as cpp_stream_compaction
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.stream_compaction cimport duplicate_keep_option
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport (
    nan_equality,
    nan_policy,
    null_equality,
    null_policy,
    size_type,
)

from cudf._lib.cpp.stream_compaction import \
    duplicate_keep_option as DuplicateKeepOption  # no-cython-lint, isort:skip

from .column cimport Column
from .table cimport Table


cpdef Table drop_nulls(Table source_table, list keys, size_type keep_threshold):
    cdef unique_ptr[table] c_result
    cdef vector[size_type] c_keys = keys
    with nogil:
        c_result = move(
            cpp_stream_compaction.drop_nulls(
                source_table.view(), c_keys, keep_threshold
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Table apply_boolean_mask(Table source_table, Column boolean_mask):
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_stream_compaction.apply_boolean_mask(
                source_table.view(), boolean_mask.view()
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef size_type distinct_count(
    Column source_table,
    null_policy null_handling,
    nan_policy nan_handling
):
    return cpp_stream_compaction.distinct_count(
        source_table.view(), null_handling, nan_handling
    )


cpdef Table stable_distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
):
    cdef unique_ptr[table] c_result
    cdef vector[size_type] c_keys = keys
    with nogil:
        c_result = move(
            cpp_stream_compaction.stable_distinct(
                input.view(), c_keys, keep, nulls_equal
            )
        )
    return Table.from_libcudf(move(c_result))


cpdef Column distinct_indices(
    Table input,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_stream_compaction.distinct_indices(
                input.view(), keep, nulls_equal, nans_equal
            )
        )
    return Column.from_libcudf(move(c_result))
