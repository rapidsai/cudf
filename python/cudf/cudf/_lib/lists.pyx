# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool

from pylibcudf.libcudf.types cimport null_order, size_type

from cudf._lib.column cimport Column
from cudf._lib.utils cimport columns_from_pylibcudf_table

import pylibcudf

from pylibcudf cimport Scalar


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
    return Column.from_pylibcudf(
        pylibcudf.lists.distinct(
            col.to_pylibcudf(mode="read"),
            nulls_equal,
            nans_all_equal,
        )
    )


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
