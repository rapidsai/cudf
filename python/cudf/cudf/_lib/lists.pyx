# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr, make_shared
from libcpp.utility cimport move

from cudf._lib.cpp.lists.count_elements cimport (
    count_elements as cpp_count_elements
)
from cudf._lib.cpp.lists.explode cimport (
    explode_outer as cpp_explode_outer
)
from cudf._lib.cpp.lists.drop_list_duplicates cimport (
    drop_list_duplicates as cpp_drop_list_duplicates
)
from cudf._lib.cpp.lists.sorting cimport (
    sort_lists as cpp_sort_lists
)
from cudf._lib.cpp.lists.combine cimport (
    concatenate_rows as cpp_concatenate_rows,
    concatenate_null_policy,
    concatenate_list_elements as cpp_concatenate_list_elements
)

from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.column.column cimport column

from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.cpp.scalar.scalar cimport scalar

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport (
    size_type,
    null_equality,
    null_policy,
    order,
    null_order,
    nan_equality
)

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table

from cudf._lib.types cimport (
    underlying_type_t_null_order, underlying_type_t_order
)
from cudf.core.dtypes import ListDtype

from cudf._lib.cpp.lists.contains cimport contains

from cudf._lib.cpp.lists.extract cimport extract_list_element


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


def explode_outer(Table tbl, int explode_column_idx, bool ignore_index=False):
    cdef table_view c_table_view = (
        tbl.data_view() if ignore_index else tbl.view()
    )
    cdef size_type c_explode_column_idx = explode_column_idx

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_explode_outer(c_table_view, c_explode_column_idx))

    return Table.from_unique_ptr(
        move(c_result),
        column_names=tbl._column_names,
        index_names=None if ignore_index else tbl._index_names
    )


def drop_list_duplicates(Column col, bool nulls_equal, bool nans_all_equal):
    """
    nans_all_equal == True indicates that libcudf should treat any two elements
    from {+nan, -nan} as equal, and as unequal otherwise.
    nulls_equal == True indicates that libcudf should treat any two nulls as
    equal, and as unequal otherwise.
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
            cpp_drop_list_duplicates(list_view.get()[0],
                                     c_nulls_equal,
                                     c_nans_equal)
        )
    return Column.from_unique_ptr(move(c_result))


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


def extract_element(Column col, size_type index):
    # shared_ptr required because lists_column_view has no default
    # ctor
    cdef shared_ptr[lists_column_view] list_view = (
        make_shared[lists_column_view](col.view())
    )

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(extract_list_element(list_view.get()[0], index))

    result = Column.from_unique_ptr(move(c_result))
    return result


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


def concatenate_rows(Table tbl):
    cdef unique_ptr[column] c_result

    cdef table_view c_table_view = tbl.view()

    with nogil:
        c_result = move(cpp_concatenate_rows(
            c_table_view,
        ))

    result = Column.from_unique_ptr(move(c_result))
    return result


def concatenate_list_elements(Column input_column, dropna=False):
    cdef concatenate_null_policy policy = (
        concatenate_null_policy.IGNORE if dropna
        else concatenate_null_policy.NULLIFY_OUTPUT_ROW
    )
    cdef column_view c_input = input_column.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_concatenate_list_elements(
            c_input,
            policy
        ))

    return Column.from_unique_ptr(move(c_result))
