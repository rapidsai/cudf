# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.lists.contains cimport (
    contains,
    index_of as cpp_index_of,
)
from cudf._lib.pylibcudf.libcudf.lists.count_elements cimport (
    count_elements as cpp_count_elements,
)
from cudf._lib.pylibcudf.libcudf.lists.extract cimport extract_list_element
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)
from cudf._lib.pylibcudf.libcudf.lists.sorting cimport (
    sort_lists as cpp_sort_lists,
)
from cudf._lib.pylibcudf.libcudf.lists.stream_compaction cimport (
    distinct as cpp_distinct,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar
from cudf._lib.pylibcudf.libcudf.types cimport (
    nan_equality,
    null_equality,
    null_order,
    order,
    size_type,
)
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport columns_from_pylibcudf_table

from cudf._lib import pylibcudf


@acquire_spill_lock()
def count_elements(Column col):

    # shared_ptr required because lists_column_view has no default
    # ctor
    cdef shared_ptr[lists_column_view] list_view = (
        make_shared[lists_column_view](col.view())
    )
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_count_elements(list_view.get()[0]))

    result = Column.from_unique_ptr(move(c_result))
    return result


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
    cdef shared_ptr[lists_column_view] list_view = (
        make_shared[lists_column_view](col.view())
    )
    cdef order c_sort_order = (
        order.ASCENDING if ascending else order.DESCENDING
    )
    cdef null_order c_null_prec = (
        null_order.BEFORE if na_position == "first" else null_order.AFTER
    )

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_sort_lists(list_view.get()[0], c_sort_order, c_null_prec)
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def extract_element_scalar(Column col, size_type index):
    return Column.from_pylibcudf(
        pylibcudf.lists.extract_list_elements(
            col.to_pylibcudf(mode="read"),
            index,
        )
    )


@acquire_spill_lock()
def extract_element_column(Column col, Column index):
    return Column.from_pylibcudf(
        pylibcudf.lists.extract_list_elements(
            col.to_pylibcudf(mode="read"),
            index.to_pylibcudf(mode="read"),
        )
    )


@acquire_spill_lock()
def contains_scalar(Column col, object py_search_key):

    cdef DeviceScalar search_key = py_search_key.device_value

    cdef shared_ptr[lists_column_view] list_view = (
        make_shared[lists_column_view](col.view())
    )
    cdef const scalar* search_key_value = search_key.get_raw_ptr()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(contains(
            list_view.get()[0],
            search_key_value[0],
        ))
    result = Column.from_unique_ptr(move(c_result))
    return result


@acquire_spill_lock()
def index_of_scalar(Column col, object py_search_key):

    cdef DeviceScalar search_key = py_search_key.device_value

    cdef shared_ptr[lists_column_view] list_view = (
        make_shared[lists_column_view](col.view())
    )
    cdef const scalar* search_key_value = search_key.get_raw_ptr()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_index_of(
            list_view.get()[0],
            search_key_value[0],
        ))
    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def index_of_column(Column col, Column search_keys):

    cdef column_view keys_view = search_keys.view()

    cdef shared_ptr[lists_column_view] list_view = (
        make_shared[lists_column_view](col.view())
    )

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_index_of(
            list_view.get()[0],
            keys_view,
        ))
    return Column.from_unique_ptr(move(c_result))


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
