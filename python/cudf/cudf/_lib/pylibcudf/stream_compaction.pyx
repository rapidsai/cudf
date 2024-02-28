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
    """Filters out rows from the input table based on the presence of nulls.

    Parameters
    ----------
    source_table : Table
        The input table to filter.
    keys : List[size_type]
        The list of column indexes to consider for null filtering.
    keep_threshold : size_type
        The minimum number of non-nulls required to keep a row.

    Returns
    -------
    Table
        A new table with rows removed based on the null count.
    """
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
    """Filters out rows from the input table based on a boolean mask.

    Parameters
    ----------
    source_table : Table
        The input table to filter.
    boolean_mask : Column
        The boolean mask to apply to the input table.

    Returns
    -------
    Table
        A new table with rows removed based on the boolean mask.
    """
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
    """Returns the number of unique elements in the input column.

    Parameters
    ----------
    source_table : Column
        The input column to count the unique elements of.
    null_handling : null_policy
        Flag to include or exclude nulls from the count.
    nan_handling : nan_policy
        Flag to include or exclude NaNs from the count.

    Returns
    -------
    size_type
        The number of unique elements in the input column.
    """
    return cpp_stream_compaction.distinct_count(
        source_table.view(), null_handling, nan_handling
    )


cpdef Table stable_distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
):
    """Get the distinct rows from the input table, preserving input order.

    Parameters
    ----------
    input : Table
        The input table to filter.
    keys : list
        The list of column indexes to consider for distinct filtering.
    keep : duplicate_keep_option
        The option to specify which rows to keep in the case of duplicates.
    nulls_equal : null_equality
        The option to specify how nulls are handled in the comparison.

    Returns
    -------
    Table
        A new table with distinct rows from the input table.
    """
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
    """Get the indices of the distinct rows from the input table.

    Parameters
    ----------
    input : Table
        The input table to filter.
    keep : duplicate_keep_option
        The option to specify which rows to keep in the case of duplicates.
    nulls_equal : null_equality
        The option to specify how nulls are handled in the comparison.
    nans_equal : nan_equality
        The option to specify how NaNs are handled in the comparison.

    Returns
    -------
    Column
        A new column with the indices of the distinct rows from the input table.
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_stream_compaction.distinct_indices(
                input.view(), keep, nulls_equal, nans_equal
            )
        )
    return Column.from_libcudf(move(c_result))
