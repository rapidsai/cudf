# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf._lib.column cimport Column
from cudf._lib.utils cimport data_from_pylibcudf_table

from cudf._lib import pylibcudf
from cudf.core.buffer import acquire_spill_lock
import cudf


@acquire_spill_lock()
def concat_columns(object columns):
    return Column.from_pylibcudf(
        pylibcudf.concatenate.concatenate(
            [col.to_pylibcudf(mode="read") for col in columns]
        )
    )


@acquire_spill_lock()
def concat_tables(object tables, bool ignore_index=False):
    if cudf.get_option("mode.pandas_compatible"):
        if not tables:
            return None

        # Get the column names and index names from the first table
        column_names = tables[0]._column_names
        index_names = None if ignore_index else tables[0]._index_names

        # Concatenate each column separately
        concatenated_columns = []
        print(column_names)
        for i in range(len(column_names)):
            columns = [table._data.columns[i] for table in tables]
            res_col = None
            #for table in tables:
            #    if res_col is None:
            #        res_col = table._data.columns[i]
            #    else:
            #        res_col = concat_columns([res_col, table._data.columns[i]])
            res_col = concat_columns(columns)
            print("43")
            del columns
            for table in tables:
                #table._data[column_names[i]] = table._data[column_names[i]].slice(0, 0)
                pass
            concatenated_columns.append(res_col.to_pylibcudf(mode="read"))

        # Concatenate index columns if not ignoring the index
        if not ignore_index:
            index_columns = []
            for i in range(len(index_names)):
                columns = [table._index._data.columns[i] for table in tables]
                index_columns.append(concat_columns(columns).to_pylibcudf(mode="read"))
            concatenated_columns = index_columns + concatenated_columns

        # Create a new table from the concatenated columns
        return data_from_pylibcudf_table(
            pylibcudf.Table(concatenated_columns),
            column_names=column_names,
            index_names=index_names
        )
    else:
        plc_tables = []
        for table in tables:
            cols = table._data.columns
            if not ignore_index:
                cols = table._index._data.columns + cols
            plc_tables.append(pylibcudf.Table([c.to_pylibcudf(mode="read") for c in cols]))

        return data_from_pylibcudf_table(
            pylibcudf.concatenate.concatenate(plc_tables),
            column_names=tables[0]._column_names,
            index_names=None if ignore_index else tables[0]._index_names
        )