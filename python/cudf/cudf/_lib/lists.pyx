# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)
from cudf._lib.pylibcudf.libcudf.lists.stream_compaction cimport (
    distinct as cpp_distinct,
)
from cudf._lib.pylibcudf.libcudf.types cimport (
    nan_equality,
    null_equality,
    null_order,
    size_type,
)
from cudf._lib.utils cimport columns_from_pylibcudf_table

from cudf._lib import pylibcudf

from cudf._lib.pylibcudf cimport Scalar


@acquire_spill_lock()
def count_elements(Column col):
    return Column.from_pylibcudf(
        pylibcudf.lists.count_elements(
            col.to_pylibcudf(mode="read"))
    )


@acquire_spill_lock()
def explode_outer(list source_columns, int explode_column_idx):
    return columns_from_pylibcudf_table(
        pylibcudf.lists.explode_outer(
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in source_columns]),
            explode_column_idx,
        )
    )


@acquire_spill_lock()
def distinct(Column col, bool nulls_equal, bool nans_all_equal):
    """
    nulls_equal == True indicates that libcudf should treat any two nulls as
    equal, and as unequal otherwise.
    nans_all_equal == True indicates that libcudf should treat any two
    elements from {-nan, +nan} as equal, and as unequal otherwise.
    """
    cdef shared_ptr[lists_column_view] list_view = (
        make_shared[lists_column_view](col.view())
    )
    cdef null_equality c_nulls_equal = (
        null_equality.EQUAL if nulls_equal else null_equality.UNEQUAL
    )
    cdef nan_equality c_nans_equal = (
        nan_equality.ALL_EQUAL if nans_all_equal else nan_equality.UNEQUAL
    )

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_distinct(list_view.get()[0],
                         c_nulls_equal,
                         c_nans_equal)
        )
    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def sort_lists(Column col, bool ascending, str na_position):
    return Column.from_pylibcudf(
        pylibcudf.lists.sort_lists(
            col.to_pylibcudf(mode="read"),
            ascending,
            null_order.BEFORE if na_position == "first" else null_order.AFTER,
            False,
        )
    )


@acquire_spill_lock()
def extract_element_scalar(Column col, size_type index):
    return Column.from_pylibcudf(
        pylibcudf.lists.extract_list_element(
            col.to_pylibcudf(mode="read"),
            index,
        )
    )


@acquire_spill_lock()
def extract_element_column(Column col, Column index):
    return Column.from_pylibcudf(
        pylibcudf.lists.extract_list_element(
            col.to_pylibcudf(mode="read"),
            index.to_pylibcudf(mode="read"),
        )
    )


@acquire_spill_lock()
def contains_scalar(Column col, py_search_key):
    return Column.from_pylibcudf(
        pylibcudf.lists.contains(
            col.to_pylibcudf(mode="read"),
            <Scalar> py_search_key.device_value.c_value,
        )
    )


@acquire_spill_lock()
def index_of_scalar(Column col, object py_search_key):
    return Column.from_pylibcudf(
        pylibcudf.lists.index_of(
            col.to_pylibcudf(mode="read"),
            <Scalar> py_search_key.device_value.c_value,
            True,
        )
    )


@acquire_spill_lock()
def index_of_column(Column col, Column search_keys):
    return Column.from_pylibcudf(
        pylibcudf.lists.index_of(
            col.to_pylibcudf(mode="read"),
            search_keys.to_pylibcudf(mode="read"),
            True,
        )
    )


@acquire_spill_lock()
def concatenate_rows(list source_columns):
    return Column.from_pylibcudf(
        pylibcudf.lists.concatenate_rows(
            pylibcudf.Table([
                c.to_pylibcudf(mode="read") for c in source_columns
            ])
        )
    )


@acquire_spill_lock()
def concatenate_list_elements(Column input_column, dropna=False):
    return Column.from_pylibcudf(
        pylibcudf.lists.concatenate_list_elements(
            input_column.to_pylibcudf(mode="read"),
            dropna,
        )
    )
