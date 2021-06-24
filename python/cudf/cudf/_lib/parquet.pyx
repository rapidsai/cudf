# Copyright (c) 2019-2021, NVIDIA CORPORATION.

# cython: boundscheck = False

import cudf
import errno
import os
import pyarrow as pa
from collections import OrderedDict

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
    is_struct_dtype,
    is_decimal_dtype,
)

from cudf._lib.utils cimport get_column_names
from cudf._lib.utils import (
    _index_level_name,
    generate_pandas_metadata,
)

from libc.stdlib cimport free
from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr, make_unique
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
    table_input_metadata,
    column_in_metadata,
    parquet_writer_options,
    write_parquet as parquet_writer,
    parquet_chunked_writer as cpp_parquet_chunked_writer,
    chunked_parquet_writer_options,
    chunked_parquet_writer_options_builder,
    merge_rowgroup_metadata as parquet_merge_metadata,
)
from cudf._lib.column cimport Column
from cudf._lib.io.utils cimport (
    make_source_info,
    make_sink_info,
    update_struct_field_names,
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
    index_col = None
    cdef map[string, string] user_data = c_out_table.metadata.user_data
    json_str = user_data[b'pandas'].decode('utf-8')
    meta = None
    if json_str != "":
        meta = json.loads(json_str)
        if 'index_columns' in meta and len(meta['index_columns']) > 0:
            index_col = meta['index_columns']
            if isinstance(index_col[0], dict) and \
                    index_col[0]['kind'] == 'range':
                is_range_index = True
            else:
                is_range_index = False
                index_col_names = OrderedDict()
                for idx_col in index_col:
                    for c in meta['columns']:
                        if c['field_name'] == idx_col:
                            index_col_names[idx_col] = c['name']
    df = cudf.DataFrame._from_table(
        Table.from_unique_ptr(
            move(c_out_table.tbl),
            column_names=column_names
        )
    )

    update_struct_field_names(df, c_out_table.metadata.schema_info)

    if df.empty and meta is not None:
        cols_dtype_map = {}
        for col in meta['columns']:
            cols_dtype_map[col['name']] = col['numpy_type']

        if not column_names:
            column_names = [o['name'] for o in meta['columns']]
            if not is_range_index and index_col in cols_dtype_map:
                column_names.remove(index_col)

        for col in column_names:
            meta_dtype = cols_dtype_map.get(col, None)
            df._data[col] = cudf.core.column.column_empty(
                row_count=0,
                dtype=np.dtype(meta_dtype)
            )

    # Set the index column
    if index_col is not None and len(index_col) > 0:
        if is_range_index:
            range_index_meta = index_col[0]
            if row_groups is not None:
                per_file_metadata = [
                    pa.parquet.read_metadata(s) for s in filepaths_or_buffers
                ]

                filtered_idx = []
                for i, file_meta in enumerate(per_file_metadata):
                    row_groups_i = []
                    start = 0
                    for row_group in range(file_meta.num_row_groups):
                        stop = start + file_meta.row_group(row_group).num_rows
                        row_groups_i.append((start, stop))
                        start = stop

                    for rg in row_groups[i]:
                        filtered_idx.append(
                            cudf.RangeIndex(
                                start=row_groups_i[rg][0],
                                stop=row_groups_i[rg][1],
                                step=range_index_meta['step']
                            )
                        )

                if len(filtered_idx) > 0:
                    idx = cudf.concat(filtered_idx)
                else:
                    idx = cudf.Index(cudf.core.column.column_empty(0))
            else:
                idx = cudf.RangeIndex(
                    start=range_index_meta['start'],
                    stop=range_index_meta['stop'],
                    step=range_index_meta['step'],
                    name=range_index_meta['name']
                )
                if skiprows is not None:
                    idx = idx[skiprows:]
                if num_rows is not None:
                    idx = idx[:num_rows]
            df.index = idx
        elif set(index_col).issubset(column_names):
            index_data = df[index_col]
            actual_index_names = list(index_col_names.values())
            if len(index_data._data) == 1:
                idx = cudf.Index(
                    index_data._data.columns[0],
                    name=actual_index_names[0]
                )
            else:
                idx = cudf.MultiIndex.from_frame(
                    index_data,
                    names=actual_index_names
                )
            df.drop(columns=index_col, inplace=True)
            df.index = idx
        else:
            if use_pandas_metadata:
                df.index.names = index_col

    return df

cpdef write_parquet(
        Table table,
        object path,
        object index=None,
        object compression="snappy",
        object statistics="ROWGROUP",
        object metadata_file_path=None,
        object int96_timestamps=False):
    """
    Cython function to call into libcudf API, see `write_parquet`.

    See Also
    --------
    cudf.io.parquet.write_parquet
    """

    # Create the write options
    cdef unique_ptr[table_input_metadata] tbl_meta

    cdef map[string, string] user_data
    cdef table_view tv
    cdef unique_ptr[cudf_io_types.data_sink] _data_sink
    cdef cudf_io_types.sink_info sink = make_sink_info(path, _data_sink)

    if index is True or (
        index is None and not isinstance(table._index, cudf.RangeIndex)
    ):
        tv = table.view()
        tbl_meta = make_unique[table_input_metadata](tv)
        for level, idx_name in enumerate(table._index.names):
            tbl_meta.get().column_metadata[level].set_name(
                str.encode(
                    _index_level_name(idx_name, level, table._column_names)
                )
            )
        num_index_cols_meta = len(table._index.names)
    else:
        tv = table.data_view()
        tbl_meta = make_unique[table_input_metadata](tv)
        num_index_cols_meta = 0

    for i, name in enumerate(table._column_names, num_index_cols_meta):
        if not isinstance(name, str):
            raise ValueError("parquet must have string column names")

        tbl_meta.get().column_metadata[i].set_name(name.encode())
        _set_col_metadata(
            table[name]._column, tbl_meta.get().column_metadata[i]
        )

    pandas_metadata = generate_pandas_metadata(table, index)
    user_data[str.encode("pandas")] = str.encode(pandas_metadata)

    # Set the table_metadata
    tbl_meta.get().user_data = user_data

    cdef cudf_io_types.compression_type comp_type = _get_comp_type(compression)
    cdef cudf_io_types.statistics_freq stat_freq = _get_stat_freq(statistics)

    cdef parquet_writer_options args
    cdef unique_ptr[vector[uint8_t]] out_metadata_c
    cdef string c_column_chunks_file_path
    cdef bool _int96_timestamps = int96_timestamps
    if metadata_file_path is not None:
        c_column_chunks_file_path = str.encode(metadata_file_path)

    # Perform write
    with nogil:
        args = move(
            parquet_writer_options.builder(sink, tv)
            .metadata(tbl_meta.get())
            .compression(comp_type)
            .stats_level(stat_freq)
            .column_chunks_file_path(c_column_chunks_file_path)
            .int96_timestamps(_int96_timestamps)
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
    cdef bool initialized
    cdef unique_ptr[cpp_parquet_chunked_writer] writer
    cdef unique_ptr[table_input_metadata] tbl_meta
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
        self.initialized = False

    def write_table(self, Table table):
        """ Writes a single table to the file """
        if not self.initialized:
            self._initialize_chunked_state(table)

        cdef table_view tv
        if self.index is not False and (
            table._index.name is not None or
                isinstance(table._index, cudf.core.multiindex.MultiIndex)):
            tv = table.view()
        else:
            tv = table.data_view()

        with nogil:
            self.writer.get()[0].write(tv)

    def close(self, object metadata_file_path=None):
        cdef unique_ptr[vector[uint8_t]] out_metadata_c
        cdef string column_chunks_file_path

        if not self.initialized:
            return None

        # Update metadata-collection options
        if metadata_file_path is not None:
            column_chunks_file_path = str.encode(metadata_file_path)

        with nogil:
            out_metadata_c = move(
                self.writer.get()[0].close(column_chunks_file_path)
            )

        if metadata_file_path is not None:
            out_metadata_py = BufferArrayFromVector.from_unique_ptr(
                move(out_metadata_c)
            )
            return np.asarray(out_metadata_py)
        return None

    def __dealloc__(self):
        self.close()

    def _initialize_chunked_state(self, Table table):
        """ Prepares all the values required to build the
        chunked_parquet_writer_options and creates a writer"""
        cdef table_view tv

        # Set the table_metadata
        num_index_cols_meta = 0
        self.tbl_meta = make_unique[table_input_metadata](table.data_view())
        if self.index is not False:
            if isinstance(table._index, cudf.core.multiindex.MultiIndex):
                tv = table.view()
                self.tbl_meta = make_unique[table_input_metadata](tv)
                for level, idx_name in enumerate(table._index.names):
                    self.tbl_meta.get().column_metadata[level].set_name(
                        (str.encode(idx_name))
                    )
                num_index_cols_meta = len(table._index.names)
            else:
                if table._index.name is not None:
                    tv = table.view()
                    self.tbl_meta = make_unique[table_input_metadata](tv)
                    self.tbl_meta.get().column_metadata[0].set_name(
                        str.encode(table._index.name)
                    )
                    num_index_cols_meta = 1

        for i, name in enumerate(table._column_names, num_index_cols_meta):
            self.tbl_meta.get().column_metadata[i].set_name(name.encode())
            _set_col_metadata(
                table[name]._column, self.tbl_meta.get().column_metadata[i]
            )

        pandas_metadata = generate_pandas_metadata(table, self.index)
        self.tbl_meta.get().user_data[str.encode("pandas")] = \
            str.encode(pandas_metadata)

        cdef chunked_parquet_writer_options args
        with nogil:
            args = move(
                chunked_parquet_writer_options.builder(self.sink)
                .metadata(self.tbl_meta.get())
                .compression(self.comp_type)
                .stats_level(self.stat_freq)
                .build()
            )
            self.writer.reset(new cpp_parquet_chunked_writer(args))
        self.initialized = True


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


cdef cudf_io_types.statistics_freq _get_stat_freq(object statistics):
    statistics = str(statistics).upper()
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


cdef _set_col_metadata(Column col, column_in_metadata& col_meta):
    if is_struct_dtype(col):
        for i, (child_col, name) in enumerate(
            zip(col.children, list(col.dtype.fields))
        ):
            col_meta.child(i).set_name(name.encode())
            _set_col_metadata(child_col, col_meta.child(i))
    elif is_list_dtype(col):
        _set_col_metadata(col.children[1], col_meta.child(1))
    else:
        if is_decimal_dtype(col):
            col_meta.set_decimal_precision(col.dtype.precision)
        return
