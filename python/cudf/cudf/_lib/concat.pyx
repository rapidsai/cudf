# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf._lib.column cimport Column
from cudf._lib.utils cimport data_from_pylibcudf_table

import pylibcudf

from cudf.core.buffer import acquire_spill_lock


@acquire_spill_lock()
def concat_columns(object columns):
    return Column.from_pylibcudf(
        pylibcudf.concatenate.concatenate(
            [col.to_pylibcudf(mode="read") for col in columns]
        )
    )


@acquire_spill_lock()
def concat_tables(object tables, bool ignore_index=False):
    plc_tables = []
    for table in tables:
        cols = table._columns
        if not ignore_index:
            cols = table._index._columns + cols
        plc_tables.append(pylibcudf.Table([c.to_pylibcudf(mode="read") for c in cols]))

    return data_from_pylibcudf_table(
        pylibcudf.concatenate.concatenate(plc_tables),
        column_names=tables[0]._column_names,
        index_names=None if ignore_index else tables[0]._index_names
    )
