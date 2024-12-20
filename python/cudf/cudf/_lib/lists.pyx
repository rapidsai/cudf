# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool

from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column
from cudf._lib.utils cimport columns_from_pylibcudf_table

import pylibcudf as plc


@acquire_spill_lock()
def count_elements(Column col):
    return Column.from_pylibcudf(
        plc.lists.count_elements(
            col.to_pylibcudf(mode="read"))
    )


@acquire_spill_lock()
def explode_outer(list source_columns, int explode_column_idx):
    return columns_from_pylibcudf_table(
        plc.lists.explode_outer(
            plc.Table([c.to_pylibcudf(mode="read") for c in source_columns]),
            explode_column_idx,
        )
    )


@acquire_spill_lock()
def distinct(Column col, bool nulls_equal, bool nans_all_equal):
    return Column.from_pylibcudf(
        plc.lists.distinct(
            col.to_pylibcudf(mode="read"),
            (
                plc.types.NullEquality.EQUAL
                if nulls_equal
                else plc.types.NullEquality.UNEQUAL
            ),
            (
                plc.types.NanEquality.ALL_EQUAL
                if nans_all_equal
                else plc.types.NanEquality.UNEQUAL
            ),
        )
    )


@acquire_spill_lock()
def sort_lists(Column col, bool ascending, str na_position):
    return Column.from_pylibcudf(
        plc.lists.sort_lists(
            col.to_pylibcudf(mode="read"),
            plc.types.Order.ASCENDING if ascending else plc.types.Order.DESCENDING,
            (
                plc.types.NullOrder.BEFORE
                if na_position == "first"
                else plc.types.NullOrder.AFTER
            ),
            False,
        )
    )


@acquire_spill_lock()
def extract_element_scalar(Column col, size_type index):
    return Column.from_pylibcudf(
        plc.lists.extract_list_element(
            col.to_pylibcudf(mode="read"),
            index,
        )
    )


@acquire_spill_lock()
def extract_element_column(Column col, Column index):
    return Column.from_pylibcudf(
        plc.lists.extract_list_element(
            col.to_pylibcudf(mode="read"),
            index.to_pylibcudf(mode="read"),
        )
    )


@acquire_spill_lock()
def contains_scalar(Column col, py_search_key):
    return Column.from_pylibcudf(
        plc.lists.contains(
            col.to_pylibcudf(mode="read"),
            py_search_key.device_value.c_value,
        )
    )


@acquire_spill_lock()
def index_of_scalar(Column col, object py_search_key):
    return Column.from_pylibcudf(
        plc.lists.index_of(
            col.to_pylibcudf(mode="read"),
            py_search_key.device_value.c_value,
            plc.lists.DuplicateFindOption.FIND_FIRST,
        )
    )


@acquire_spill_lock()
def index_of_column(Column col, Column search_keys):
    return Column.from_pylibcudf(
        plc.lists.index_of(
            col.to_pylibcudf(mode="read"),
            search_keys.to_pylibcudf(mode="read"),
            plc.lists.DuplicateFindOption.FIND_FIRST,
        )
    )


@acquire_spill_lock()
def concatenate_rows(list source_columns):
    return Column.from_pylibcudf(
        plc.lists.concatenate_rows(
            plc.Table([
                c.to_pylibcudf(mode="read") for c in source_columns
            ])
        )
    )


@acquire_spill_lock()
def concatenate_list_elements(Column input_column, dropna=False):
    return Column.from_pylibcudf(
        plc.lists.concatenate_list_elements(
            input_column.to_pylibcudf(mode="read"),
            plc.lists.ConcatenateNullPolicy.IGNORE
            if dropna
            else plc.lists.ConcatenateNullPolicy.NULLIFY_OUTPUT_ROW,
        )
    )


@acquire_spill_lock()
def segmented_gather(Column source_column, Column gather_map):
    return Column.from_pylibcudf(
        plc.lists.segmented_gather(
            source_column.to_pylibcudf(mode="read"),
            gather_map.to_pylibcudf(mode="read"),
        )
    )
