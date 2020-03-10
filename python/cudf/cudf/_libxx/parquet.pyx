# Copyright (c) 2019-2020, NVIDIA CORPORATION.

# cython: boundscheck = False

import cudf
import errno
import os
import pyarrow as pa
import json

from libc.stdlib cimport free
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector

from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.table cimport Table
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport (
    table_view
)
from cudf._libxx.move cimport move
from cudf._libxx.cpp.io.functions cimport (
    write_parquet_args,
    write_parquet as parquet_writer,
    read_parquet_args,
    read_parquet as parquet_reader
)
from cudf._libxx.io.utils cimport (
    make_source_info
)

cimport cudf._libxx.cpp.types as cudf_types
cimport cudf._libxx.cpp.io.types as cudf_io_types

cpdef generate_pandas_metadata(Table table, index):
    col_names = []
    types = []
    index_levels = []
    index_descriptors = []

    # Columns
    for name, col in table._data.items():
        col_names.append(name)
        types.append(col.to_arrow().type)

    # Indexes
    if index is not False:
        for name in table._index.names:
            if name is not None:
                if isinstance(table._index, cudf.core.multiindex.MultiIndex):
                    idx = table.index.get_level_values(name)
                else:
                    idx = table.index

                if isinstance(idx, cudf.core.index.RangeIndex):
                    descr = {
                        "kind": "range",
                        "name": table.index.name,
                        "start": table.index._start,
                        "stop": table.index._stop,
                        "step": 1,
                    }
                else:
                    index_arrow = idx.to_arrow()
                    descr = name
                    types.append(index_arrow.type)
                    col_names.append(name)
                    index_levels.append(idx)
                index_descriptors.append(descr)
            else:
                col_names.append(name)

    metadata = pa.pandas_compat.construct_metadata(
        table,
        col_names,
        index_levels,
        index_descriptors,
        index,
        types,
    )

    md = metadata[b'pandas']
    json_str = md.decode("utf-8")
    return json_str

cpdef read_parquet(filepath_or_buffer, columns=None, row_group=None,
                   row_group_count=None, skip_rows=None, num_rows=None,
                   strings_to_categorical=False, use_pandas_metadata=False):
    """
    Cython function to call into libcudf API, see `read_parquet`.

    See Also
    --------
    cudf.io.parquet.read_parquet
    cudf.io.parquet.to_parquet
    """

    cdef cudf_io_types.source_info source = make_source_info(
        filepath_or_buffer)

    # Setup parquet reader arguments
    cdef read_parquet_args args = read_parquet_args(source)

    if columns is not None:
        args.columns.reserve(len(columns))
        for col in columns or []:
            args.columns.push_back(str(col).encode())
    args.strings_to_categorical = strings_to_categorical
    args.use_pandas_metadata = use_pandas_metadata

    args.skip_rows = skip_rows if skip_rows is not None else 0
    args.num_rows = num_rows if num_rows is not None else -1
    args.row_group = row_group if row_group is not None else -1
    args.row_group_count = row_group_count \
        if row_group_count is not None else -1
    args.timestamp_type = cudf_types.data_type(cudf_types.type_id.EMPTY)

    # Read Parquet
    cdef cudf_io_types.table_with_metadata c_out_table

    with nogil:
        c_out_table = move(parquet_reader(args))

    column_names = [x.decode() for x in c_out_table.metadata.column_names]

    # Access the Parquet user_data json to find the index
    index_col = None
    cdef map[string, string] user_data = c_out_table.metadata.user_data
    json_str = user_data[b'pandas'].decode('utf-8')
    if json_str != "":
        meta = json.loads(json_str)
        if 'index_columns' in meta and len(meta['index_columns']) > 0:
            index_col = meta['index_columns'][0]

    df = cudf.DataFrame._from_table(
        Table.from_unique_ptr(move(c_out_table.tbl),
                              column_names=column_names)
    )

    # if index_col is not None and index_col in column_names:
    if index_col is not None and index_col in column_names:
        df = df.set_index(index_col)
        new_index_name = pa.pandas_compat._backwards_compatible_index_name(
            df.index.name, df.index.name
        )
        df.index.name = new_index_name
    else:
        df.index.name = index_col

    return df

cpdef write_parquet(
        Table table,
        path,
        index=None,
        compression=None,
        statistics="ROWGROUP"):
    """
    Cython function to call into libcudf API, see `write_parquet`.

    See Also
    --------
    cudf.io.parquet.write_parquet
    """

    # Create the write options
    cdef string filepath = <string>str(path).encode()
    cdef cudf_io_types.sink_info sink = cudf_io_types.sink_info(filepath)
    cdef unique_ptr[cudf_io_types.table_metadata] tbl_meta = \
        make_unique[cudf_io_types.table_metadata]()

    cdef vector[string] column_names
    cdef map[string, string] user_data
    cdef table_view tv = table.data_view()

    if index is not False:
        tv = table.view()
        if isinstance(table._index, cudf.core.multiindex.MultiIndex):
            for idx_name in table._index.names:
                column_names.push_back(str.encode(idx_name))
        else:
            if table._index.name is not None:
                column_names.push_back(str.encode(table._index.name))
            else:
                # No named index exists so just write out columns
                tv = table.data_view()

    for col_name in table._column_names:
        column_names.push_back(str.encode(col_name))

    pandas_metadata = generate_pandas_metadata(table, index)
    user_data[str.encode("pandas")] = str.encode(pandas_metadata)

    # Set the table_metadata
    tbl_meta.get().column_names = column_names
    tbl_meta.get().user_data = user_data

    cdef cudf_io_types.compression_type comp_type
    if compression is None:
        comp_type = cudf_io_types.compression_type.NONE
    elif compression == "snappy":
        comp_type = cudf_io_types.compression_type.SNAPPY
    else:
        raise ValueError("Unsupported `compression` type")

    cdef cudf_io_types.statistics_freq stat_freq
    statistics = statistics.upper()
    if statistics == "NONE":
        stat_freq = cudf_io_types.statistics_freq.STATISTICS_NONE
    elif statistics == "ROWGROUP":
        stat_freq = cudf_io_types.statistics_freq.STATISTICS_ROWGROUP
    elif statistics == "PAGE":
        stat_freq = cudf_io_types.statistics_freq.STATISTICS_PAGE
    else:
        raise ValueError("Unsupported `statistics_freq` type")

    cdef write_parquet_args args

    # Perform write
    with nogil:
        args = write_parquet_args(sink,
                                  tv,
                                  tbl_meta.get(),
                                  comp_type,
                                  stat_freq)
        parquet_writer(args)
