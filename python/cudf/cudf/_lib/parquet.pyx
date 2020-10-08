# Copyright (c) 2019-2020, NVIDIA CORPORATION.

# cython: boundscheck = False

import cudf
import errno
import os
import pyarrow as pa

try:
    import ujson as json
except ImportError:
    import json

from cython.operator import dereference
import numpy as np

from cudf.utils.dtypes import (
    np_to_pa_dtype,
    is_categorical_dtype,
    is_list_dtype,
    is_struct_dtype
)
from libc.stdlib cimport free
from libc.stdint cimport uint8_t
from libcpp.memory cimport shared_ptr, unique_ptr, make_unique
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.utility cimport move
from libcpp cimport bool


from cudf._lib.cpp.types cimport data_type, size_type
from cudf._lib.table cimport Table
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport (
    table_view
)
from cudf._lib.cpp.io.parquet cimport (
    read_parquet as parquet_reader,
    parquet_reader_options,
    parquet_writer_options,
    write_parquet as parquet_writer,
    chunked_parquet_writer_options,
    chunked_parquet_writer_options_builder,
    write_parquet_chunked_begin,
    write_parquet_chunked,
    write_parquet_chunked_end,
    merge_rowgroup_metadata as parquet_merge_metadata,
    pq_chunked_state
)
from cudf._lib.column cimport Column
from cudf._lib.io.utils cimport (
    make_source_info,
    make_sink_info
)

cimport cudf._lib.cpp.types as cudf_types
cimport cudf._lib.cpp.io.types as cudf_io_types

cdef class BufferArrayFromVector:
    cdef Py_ssize_t length
    cdef unique_ptr[vector[uint8_t]] in_vec

    # these two things declare part of the buffer interface
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    @staticmethod
    cdef BufferArrayFromVector from_unique_ptr(
        unique_ptr[vector[uint8_t]] in_vec
    ):
        cdef BufferArrayFromVector buf = BufferArrayFromVector()
        buf.in_vec = move(in_vec)
        buf.length = dereference(buf.in_vec).size()
        return buf

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(uint8_t)

        self.shape[0] = self.length
        self.strides[0] = 1

        buffer.buf = dereference(self.in_vec).data()

        buffer.format = NULL  # byte
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.length * itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cpdef generate_pandas_metadata(Table table, index):
    col_names = []
    types = []
    index_levels = []
    index_descriptors = []

    # Columns
    for name, col in table._data.items():
        col_names.append(name)
        if is_categorical_dtype(col):
            raise ValueError(
                "'category' column dtypes are currently not "
                + "supported by the gpu accelerated parquet writer"
            )
        elif is_list_dtype(col):
            types.append(col.dtype.to_arrow())
        else:
            types.append(np_to_pa_dtype(col.dtype))

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
                    descr = name
                    col_names.append(name)
                    if is_categorical_dtype(idx):
                        raise ValueError(
                            "'category' column dtypes are currently not "
                            + "supported by the gpu accelerated parquet writer"
                        )
                    elif is_list_dtype(col):
                        types.append(col.dtype.to_arrow())
                    else:
                        types.append(np_to_pa_dtype(idx.dtype))
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

    md_dict = json.loads(metadata[b"pandas"])

    # correct metadata for list and struct types
    for col_meta in md_dict["columns"]:
        if col_meta["numpy_type"] in ("list", "struct"):
            col_meta["numpy_type"] = "object"

    return json.dumps(md_dict)


cpdef read_parquet(filepaths_or_buffers, columns=None, row_groups=None,
                   skiprows=None, num_rows=None, strings_to_categorical=False,
                   use_pandas_metadata=True):
    """
    Cython function to call into libcudf API, see `read_parquet`.

    See Also
    --------
    cudf.io.parquet.read_parquet
    cudf.io.parquet.to_parquet
    """

    cdef cudf_io_types.source_info source = make_source_info(
        filepaths_or_buffers)

    cdef vector[string] cpp_columns
    cdef bool cpp_strings_to_categorical = strings_to_categorical
    cdef bool cpp_use_pandas_metadata = use_pandas_metadata
    cdef size_type cpp_skiprows = skiprows if skiprows is not None else 0
    cdef size_type cpp_num_rows = num_rows if num_rows is not None else -1
    cdef vector[vector[size_type]] cpp_row_groups
    cdef data_type cpp_timestamp_type = cudf_types.data_type(
        cudf_types.type_id.EMPTY
    )

    if columns is not None:
        cpp_columns.reserve(len(columns))
        for col in columns or []:
            cpp_columns.push_back(str(col).encode())
    if row_groups is not None:
        cpp_row_groups = row_groups

    cdef parquet_reader_options args
    # Setup parquet reader arguments
    args = move(
        parquet_reader_options.builder(source)
        .columns(cpp_columns)
        .row_groups(cpp_row_groups)
        .convert_strings_to_categories(cpp_strings_to_categorical)
        .use_pandas_metadata(cpp_use_pandas_metadata)
        .skip_rows(cpp_skiprows)
        .num_rows(cpp_num_rows)
        .timestamp_type(cpp_timestamp_type)
        .build()
    )

    # Read Parquet
    cdef cudf_io_types.table_with_metadata c_out_table

    with nogil:
        c_out_table = move(parquet_reader(args))

    column_names = [x.decode() for x in c_out_table.metadata.column_names]

    # Access the Parquet user_data json to find the index
    index_col = ''
    cdef map[string, string] user_data = c_out_table.metadata.user_data
    json_str = user_data[b'pandas'].decode('utf-8')
    meta = None
    if json_str != "":
        meta = json.loads(json_str)
        if 'index_columns' in meta and len(meta['index_columns']) > 0:
            index_col = meta['index_columns'][0]

    df = cudf.DataFrame._from_table(
        Table.from_unique_ptr(
            move(c_out_table.tbl),
            column_names=column_names
        )
    )

    _update_struct_field_names(df, c_out_table.metadata.schema_info)

    if df.empty and meta is not None:
        cols_dtype_map = {}
        for col in meta['columns']:
            cols_dtype_map[col['name']] = col['numpy_type']

        if not column_names:
            column_names = [o['name'] for o in meta['columns']]
            if index_col in cols_dtype_map:
                column_names.remove(index_col)

        for col in column_names:
            meta_dtype = cols_dtype_map.get(col, None)
            df._data[col] = cudf.core.column.column_empty(
                row_count=0,
                dtype=np.dtype(meta_dtype)
            )

    # Set the index column
    if index_col is not '' and isinstance(index_col, str):
        if index_col in column_names:
            df = df.set_index(index_col)
            new_index_name = pa.pandas_compat._backwards_compatible_index_name(
                df.index.name, df.index.name
            )
            df.index.name = new_index_name
        else:
            if use_pandas_metadata:
                df.index.name = index_col

    return df

cpdef write_parquet(
        Table table,
        object path,
        object index=None,
        object compression=None,
        str statistics="ROWGROUP",
        object metadata_file_path=None):
    """
    Cython function to call into libcudf API, see `write_parquet`.

    See Also
    --------
    cudf.io.parquet.write_parquet
    """

    # Create the write options
    cdef unique_ptr[cudf_io_types.table_metadata] tbl_meta = \
        make_unique[cudf_io_types.table_metadata]()

    cdef vector[string] column_names
    cdef map[string, string] user_data
    cdef table_view tv = table.data_view()
    cdef unique_ptr[cudf_io_types.data_sink] _data_sink
    cdef cudf_io_types.sink_info sink = make_sink_info(path, _data_sink)

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

    cdef cudf_io_types.compression_type comp_type = _get_comp_type(compression)
    cdef cudf_io_types.statistics_freq stat_freq = _get_stat_freq(statistics)

    cdef parquet_writer_options args
    cdef unique_ptr[vector[uint8_t]] out_metadata_c
    cdef string c_column_chunks_file_path
    cdef bool return_filemetadata = False
    if metadata_file_path is not None:
        c_column_chunks_file_path = str.encode(metadata_file_path)
        return_filemetadata = True

    # Perform write
    with nogil:
        args = move(
            parquet_writer_options.builder(sink, tv)
            .metadata(tbl_meta.get())
            .compression(comp_type)
            .stats_level(stat_freq)
            .column_chunks_file_path(c_column_chunks_file_path)
            .return_filemetadata(return_filemetadata)
            .build()
        )
        out_metadata_c = move(parquet_writer(args))

    if metadata_file_path is not None:
        out_metadata_py = BufferArrayFromVector.from_unique_ptr(
            move(out_metadata_c)
        )
        return np.asarray(out_metadata_py)
    else:
        return None


cdef class ParquetWriter:
    """
    ParquetWriter lets you incrementally write out a Parquet file from a series
    of cudf tables

    See Also
    --------
    cudf.io.parquet.write_parquet
    """
    cdef shared_ptr[pq_chunked_state] state
    cdef cudf_io_types.sink_info sink
    cdef unique_ptr[cudf_io_types.data_sink] _data_sink
    cdef cudf_io_types.statistics_freq stat_freq
    cdef cudf_io_types.compression_type comp_type
    cdef object index

    def __cinit__(self, object path, object index=None,
                  object compression=None, str statistics="ROWGROUP"):
        self.sink = make_sink_info(path, self._data_sink)
        self.stat_freq = _get_stat_freq(statistics)
        self.comp_type = _get_comp_type(compression)
        self.index = index

    def write_table(self, Table table):
        """ Writes a single table to the file """
        if not self.state:
            self._initialize_chunked_state(table)

        cdef table_view tv = table.data_view()
        if self.index is not False:
            if isinstance(table._index, cudf.core.multiindex.MultiIndex) \
                    or table._index.name is not None:
                tv = table.view()

        with nogil:
            write_parquet_chunked(tv, self.state)

    def close(self, object metadata_file_path=None):
        cdef unique_ptr[vector[uint8_t]] out_metadata_c
        cdef bool return_meta
        cdef string column_chunks_file_path

        if not self.state:
            return None

        # Update metadata-collection options
        if metadata_file_path is not None:
            column_chunks_file_path = str.encode(metadata_file_path)
            return_meta = True
        else:
            return_meta = False

        with nogil:
            out_metadata_c = move(
                write_parquet_chunked_end(
                    self.state, return_meta, column_chunks_file_path
                )
            )
            self.state.reset()

        if metadata_file_path is not None:
            out_metadata_py = BufferArrayFromVector.from_unique_ptr(
                move(out_metadata_c)
            )
            return np.asarray(out_metadata_py)
        return None

    def __dealloc__(self):
        self.close()

    def _initialize_chunked_state(self, Table table):
        """ Wraps write_parquet_chunked_begin. This is called lazily on the first
        call to write, so that we can get metadata from the first table """
        cdef unique_ptr[cudf_io_types.table_metadata_with_nullability] tbl_meta
        tbl_meta = make_unique[cudf_io_types.table_metadata_with_nullability]()

        # Set the table_metadata
        tbl_meta.get().column_names = _get_column_names(table, self.index)
        pandas_metadata = generate_pandas_metadata(table, self.index)
        tbl_meta.get().user_data[str.encode("pandas")] = \
            str.encode(pandas_metadata)

        # call write_parquet_chunked_begin
        cdef chunked_parquet_writer_options args
        with nogil:
            args = move(
                chunked_parquet_writer_options.builder(self.sink)
                .nullable_metadata(tbl_meta.get())
                .compression(self.comp_type)
                .stats_level(self.stat_freq)
                .build()
            )
            self.state = write_parquet_chunked_begin(args)


cpdef merge_filemetadata(object filemetadata_list):
    """
    Cython function to call into libcudf API, see `merge_rowgroup_metadata`.

    See Also
    --------
    cudf.io.parquet.merge_rowgroup_metadata
    """
    cdef vector[unique_ptr[vector[uint8_t]]] list_c
    cdef vector[uint8_t] blob_c
    cdef unique_ptr[vector[uint8_t]] output_c

    for blob_py in filemetadata_list:
        blob_c = blob_py
        list_c.push_back(make_unique[vector[uint8_t]](blob_c))

    with nogil:
        output_c = move(parquet_merge_metadata(list_c))

    out_metadata_py = BufferArrayFromVector.from_unique_ptr(move(output_c))
    return np.asarray(out_metadata_py)


cdef cudf_io_types.statistics_freq _get_stat_freq(str statistics):
    statistics = statistics.upper()
    if statistics == "NONE":
        return cudf_io_types.statistics_freq.STATISTICS_NONE
    elif statistics == "ROWGROUP":
        return cudf_io_types.statistics_freq.STATISTICS_ROWGROUP
    elif statistics == "PAGE":
        return cudf_io_types.statistics_freq.STATISTICS_PAGE
    else:
        raise ValueError("Unsupported `statistics_freq` type")


cdef cudf_io_types.compression_type _get_comp_type(object compression):
    if compression is None:
        return cudf_io_types.compression_type.NONE
    elif compression == "snappy":
        return cudf_io_types.compression_type.SNAPPY
    else:
        raise ValueError("Unsupported `compression` type")


cdef vector[string] _get_column_names(Table table, object index):
    cdef vector[string] column_names
    if index is not False:
        if isinstance(table._index, cudf.core.multiindex.MultiIndex):
            for idx_name in table._index.names:
                column_names.push_back(str.encode(idx_name))
        else:
            if table._index.name is not None:
                column_names.push_back(str.encode(table._index.name))

    for col_name in table._column_names:
        column_names.push_back(str.encode(col_name))

    return column_names


cdef _update_struct_field_names(
    Table table,
    vector[cudf_io_types.column_name_info]& schema_info
):
    for i, (name, col) in enumerate(table._data.items()):
        table._data[name] = _update_column_struct_field_names(
            col, schema_info[i]
        )

cdef Column _update_column_struct_field_names(
    Column col,
    cudf_io_types.column_name_info& info
):
    cdef vector[string] field_names

    if is_struct_dtype(col):
        field_names.reserve(len(col.base_children))
        for i in range(info.children.size()):
            field_names.push_back(info.children[i].name)
        col = col._rename_fields(
            field_names
        )

    if col.children:
        children = list(col.children)
        for i, child in enumerate(children):
            children[i] = _update_column_struct_field_names(
                child,
                info.children[i]
            )
        col.set_base_children(tuple(children))
    return col
