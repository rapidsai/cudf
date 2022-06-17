
# Copyright (c) 2021, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.dictionary.dictionary_column_view cimport (
    dictionary_column_view,
)
from cudf._lib.cpp.dictionary.update_keys cimport (
    add_keys,
    remove_keys,
    set_keys,
)

from cudf.core.column.column import arange

from cudf._lib.column cimport Column


def set_categories(Column category_column, Column categories):
    cdef column_view keys_view = categories.view()

    cdef shared_ptr[dictionary_column_view] dict_view = (
        make_shared[dictionary_column_view](category_column.view())
    )
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(set_keys(dict_view.get()[0], keys_view))

    return Column.from_unique_ptr(move(c_result))


def add_categories(Column category_column, Column new_categories):
    cdef column_view keys_view = new_categories.view()

    cdef shared_ptr[dictionary_column_view] dict_view = (
        make_shared[dictionary_column_view](category_column.view())
    )
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(add_keys(dict_view.get()[0], keys_view))

    return Column.from_unique_ptr(move(c_result))


def remove_categories(Column category_column, Column categories_to_remove):
    cdef column_view keys_view = categories_to_remove.view()

    cdef shared_ptr[dictionary_column_view] dict_view = (
        make_shared[dictionary_column_view](category_column.view())
    )
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(remove_keys(dict_view.get()[0], keys_view))

    return Column.from_unique_ptr(move(c_result))
