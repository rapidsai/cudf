# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import io

import pyarrow as pa

import cudf
from cudf.core.buffer import acquire_spill_lock

try:
    import ujson as json
except ImportError:
    import json

import numpy as np

from cython.operator cimport dereference

from cudf.api.types import is_list_like

from cudf._lib.utils cimport data_from_unique_ptr

from cudf._lib.utils import _index_level_name, generate_pandas_metadata

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector

cimport cudf._lib.pylibcudf.libcudf.io.data_sink as cudf_io_data_sink
cimport cudf._lib.pylibcudf.libcudf.io.types as cudf_io_types
cimport cudf._lib.pylibcudf.libcudf.types as cudf_types
from cudf._lib.column cimport Column
from cudf._lib.expressions cimport Expression
from cudf._lib.io.datasource cimport NativeFileDatasource
from cudf._lib.io.utils cimport (
    make_sinks_info,
    make_source_info,
    update_struct_field_names,
)
from cudf._lib.pylibcudf.libcudf.expressions cimport expression
from cudf._lib.pylibcudf.libcudf.io.parquet cimport (
    chunked_parquet_writer_options,
    merge_row_group_metadata as parquet_merge_metadata,
    parquet_chunked_writer as cpp_parquet_chunked_writer,
    parquet_reader_options,
    parquet_reader_options_builder,
    parquet_writer_options,
    read_parquet as parquet_reader,
    write_parquet as parquet_writer,
)
from cudf._lib.pylibcudf.libcudf.io.parquet_metadata cimport (
    parquet_metadata,
    read_parquet_metadata as parquet_metadata_reader,
)
from cudf._lib.pylibcudf.libcudf.io.types cimport (
    column_in_metadata,
    table_input_metadata,
)
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport data_type, size_type
from cudf._lib.utils cimport table_view_from_table

from pyarrow.lib import NativeFile

from cudf.utils.ioutils import _ROW_GROUP_SIZE_BYTES_DEFAULT


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


def _parse_metadata(meta):
    file_is_range_index = False
    file_index_cols = None
    file_column_dtype = None

    if 'index_columns' in meta and len(meta['index_columns']) > 0:
        file_index_cols = meta['index_columns']

        if isinstance(file_index_cols[0], dict) and \
                file_index_cols[0]['kind'] == 'range':
            file_is_range_index = True
    if 'column_indexes' in meta and len(meta['column_indexes']) == 1:
        file_column_dtype = meta['column_indexes'][0]["numpy_type"]
    return file_is_range_index, file_index_cols, file_column_dtype


cpdef read_parquet(filepaths_or_buffers, columns=None, row_groups=None,
                   use_pandas_metadata=True,
                   Expression filters=None):
    """
    Cython function to call into libcudf API, see `read_parquet`.

    filters, if not None, should be an Expression that evaluates to a
    boolean predicate as a function of columns being read.

    See Also
    --------
    cudf.io.parquet.read_parquet
    cudf.io.parquet.to_parquet
    """

    # Convert NativeFile buffers to NativeFileDatasource,
    # but save original buffers in case we need to use
    # pyarrow for metadata processing
    # (See: https://github.com/rapidsai/cudf/issues/9599)
    pa_buffers = []
    for i, datasource in enumerate(filepaths_or_buffers):
        if isinstance(datasource, NativeFile):
            pa_buffers.append(datasource)
            filepaths_or_buffers[i] = NativeFileDatasource(datasource)

    cdef cudf_io_types.source_info source = make_source_info(
        filepaths_or_buffers)

    cdef bool cpp_use_pandas_metadata = use_pandas_metadata

    cdef vector[vector[size_type]] cpp_row_groups
    cdef data_type cpp_timestamp_type = cudf_types.data_type(
        cudf_types.type_id.EMPTY
    )
    if row_groups is not None:
        cpp_row_groups = row_groups

    # Setup parquet reader arguments
    cdef parquet_reader_options args
    cdef parquet_reader_options_builder builder
    builder = (
        parquet_reader_options.builder(source)
        .row_groups(cpp_row_groups)
        .use_pandas_metadata(cpp_use_pandas_metadata)
        .timestamp_type(cpp_timestamp_type)
    )
    if filters is not None:
        builder = builder.filter(<expression &>dereference(filters.c_obj.get()))

    args = move(builder.build())
    cdef vector[string] cpp_columns
    allow_range_index = True
    if columns is not None:
        cpp_columns.reserve(len(columns))
        allow_range_index = len(columns) > 0
        for col in columns:
            cpp_columns.push_back(str(col).encode())
        args.set_columns(cpp_columns)
    # Filters don't handle the range index correctly
    allow_range_index &= filters is None

    # Read Parquet
    cdef cudf_io_types.table_with_metadata c_result

    with nogil:
        c_result = move(parquet_reader(args))

    names = [info.name.decode() for info in c_result.metadata.schema_info]

    # Access the Parquet per_file_user_data to find the index
    index_col = None
    cdef vector[unordered_map[string, string]] per_file_user_data = \
        c_result.metadata.per_file_user_data

    column_index_type = None
    index_col_names = None
    is_range_index = True
    for single_file in per_file_user_data:
        json_str = single_file[b'pandas'].decode('utf-8')
        meta = None
        if json_str != "":
            meta = json.loads(json_str)
            file_is_range_index, index_col, column_index_type = _parse_metadata(meta)
            is_range_index &= file_is_range_index

            if not file_is_range_index and index_col is not None \
                    and index_col_names is None:
                index_col_names = {}
                for idx_col in index_col:
                    for c in meta['columns']:
                        if c['field_name'] == idx_col:
                            index_col_names[idx_col] = c['name']

    df = cudf.DataFrame._from_data(*data_from_unique_ptr(
        move(c_result.tbl),
        column_names=names
    ))

    update_struct_field_names(df, c_result.metadata.schema_info)

    if meta is not None:
        # Book keep each column metadata as the order
        # of `meta["columns"]` and `column_names` are not
        # guaranteed to be deterministic and same always.
        meta_data_per_column = {
            col_meta['name']: col_meta for col_meta in meta["columns"]
        }

        # update the decimal precision of each column
        for col in names:
            if isinstance(df._data[col].dtype, cudf.core.dtypes.DecimalDtype):
                df._data[col].dtype.precision = (
                    meta_data_per_column[col]["metadata"]["precision"]
                )

    # Set the index column
    if index_col is not None and len(index_col) > 0:
        if is_range_index:
            if not allow_range_index:
                return df

            if len(per_file_user_data) > 1:
                range_index_meta = {
                    "kind": "range",
                    "name": None,
                    "start": 0,
                    "stop": len(df),
                    "step": 1
                }
            else:
                range_index_meta = index_col[0]

            if row_groups is not None:
                per_file_metadata = [
                    pa.parquet.read_metadata(
                        # Pyarrow cannot read directly from bytes
                        io.BytesIO(s) if isinstance(s, bytes) else s
                    ) for s in (
                        pa_buffers or filepaths_or_buffers
                    )
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

            df._index = idx
        elif set(index_col).issubset(names):
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
            df._index = idx
        else:
            if use_pandas_metadata:
                df.index.names = index_col

    # Set column dtype for empty types.
    if len(df._data.names) == 0 and column_index_type is not None:
        df._data.label_dtype = cudf.dtype(column_index_type)
    return df

cpdef read_parquet_metadata(filepaths_or_buffers):
    """
    Cython function to call into libcudf API, see `read_parquet_metadata`.

    See Also
    --------
    cudf.io.parquet.read_parquet
    cudf.io.parquet.to_parquet
    """
    # Convert NativeFile buffers to NativeFileDatasource
    for i, datasource in enumerate(filepaths_or_buffers):
        if isinstance(datasource, NativeFile):
            filepaths_or_buffers[i] = NativeFileDatasource(datasource)

    cdef cudf_io_types.source_info source = make_source_info(filepaths_or_buffers)

    args = move(source)

    cdef parquet_metadata c_result

    # Read Parquet metadata
    with nogil:
        c_result = move(parquet_metadata_reader(args))

    # access and return results
    num_rows = c_result.num_rows()
    num_rowgroups = c_result.num_rowgroups()

    # extract row group metadata and sanitize keys
    row_group_metadata = [{k.decode(): v for k, v in metadata}
                          for metadata in c_result.rowgroup_metadata()]

    # read all column names including index column, if any
    col_names = [info.name().decode() for info in c_result.schema().root().children()]

    # access the Parquet file_footer to find the index
    index_col = None
    cdef unordered_map[string, string] file_footer = c_result.metadata()

    # get index column name(s)
    index_col_names = None
    json_str = file_footer[b'pandas'].decode('utf-8')
    meta = None
    if json_str != "":
        meta = json.loads(json_str)
        file_is_range_index, index_col, _ = _parse_metadata(meta)
        if not file_is_range_index and index_col is not None \
                and index_col_names is None:
            index_col_names = {}
            for idx_col in index_col:
                for c in meta['columns']:
                    if c['field_name'] == idx_col:
                        index_col_names[idx_col] = c['name']

    # remove the index column from the list of column names
    # only if index_col_names is not None
    if index_col_names is not None:
        col_names = [name for name in col_names if name not in index_col_names]

    # num_columns = length of list(col_names)
    num_columns = len(col_names)

    # return the metadata
    return num_rows, num_rowgroups, col_names, num_columns, row_group_metadata


@acquire_spill_lock()
def write_parquet(
    table,
    object filepaths_or_buffers,
    object index=None,
    object compression="snappy",
    object statistics="ROWGROUP",
    object metadata_file_path=None,
    object int96_timestamps=False,
    object row_group_size_bytes=_ROW_GROUP_SIZE_BYTES_DEFAULT,
    object row_group_size_rows=None,
    object max_page_size_bytes=None,
    object max_page_size_rows=None,
    object max_dictionary_size=None,
    object partitions_info=None,
    object force_nullable_schema=False,
    header_version="1.0",
    use_dictionary=True,
):
    """
    Cython function to call into libcudf API, see `write_parquet`.

    See Also
    --------
    cudf.io.parquet.write_parquet
    """

    # Create the write options
    cdef table_input_metadata tbl_meta

    cdef vector[map[string, string]] user_data
    cdef table_view tv
    cdef vector[unique_ptr[cudf_io_data_sink.data_sink]] _data_sinks
    cdef cudf_io_types.sink_info sink = make_sinks_info(
        filepaths_or_buffers, _data_sinks
    )

    if index is True or (
        index is None and not isinstance(table._index, cudf.RangeIndex)
    ):
        tv = table_view_from_table(table)
        tbl_meta = table_input_metadata(tv)
        for level, idx_name in enumerate(table._index.names):
            tbl_meta.column_metadata[level].set_name(
                str.encode(
                    _index_level_name(idx_name, level, table._column_names)
                )
            )
        num_index_cols_meta = len(table._index.names)
    else:
        tv = table_view_from_table(table, ignore_index=True)
        tbl_meta = table_input_metadata(tv)
        num_index_cols_meta = 0

    for i, name in enumerate(table._column_names, num_index_cols_meta):
        if not isinstance(name, str):
            if cudf.get_option("mode.pandas_compatible"):
                tbl_meta.column_metadata[i].set_name(str(name).encode())
            else:
                raise ValueError(
                    "Writing a Parquet file requires string column names"
                )
        else:
            tbl_meta.column_metadata[i].set_name(name.encode())

        _set_col_metadata(
            table[name]._column,
            tbl_meta.column_metadata[i],
            force_nullable_schema
        )

    cdef map[string, string] tmp_user_data
    if partitions_info is not None:
        for start_row, num_row in partitions_info:
            partitioned_df = table.iloc[start_row: start_row + num_row].copy(
                deep=False
            )
            pandas_metadata = generate_pandas_metadata(partitioned_df, index)
            tmp_user_data[str.encode("pandas")] = str.encode(pandas_metadata)
            user_data.push_back(tmp_user_data)
            tmp_user_data.clear()
    else:
        pandas_metadata = generate_pandas_metadata(table, index)
        tmp_user_data[str.encode("pandas")] = str.encode(pandas_metadata)
        user_data.push_back(tmp_user_data)

    if header_version not in ("1.0", "2.0"):
        raise ValueError(
            f"Invalid parquet header version: {header_version}. "
            "Valid values are '1.0' and '2.0'"
        )

    dict_policy = (
        cudf_io_types.dictionary_policy.ADAPTIVE
        if use_dictionary
        else cudf_io_types.dictionary_policy.NEVER
    )

    cdef cudf_io_types.compression_type comp_type = _get_comp_type(compression)
    cdef cudf_io_types.statistics_freq stat_freq = _get_stat_freq(statistics)

    cdef unique_ptr[vector[uint8_t]] out_metadata_c
    cdef vector[string] c_column_chunks_file_paths
    cdef bool _int96_timestamps = int96_timestamps
    cdef vector[cudf_io_types.partition_info] partitions

    # Perform write
    cdef parquet_writer_options args = move(
        parquet_writer_options.builder(sink, tv)
        .metadata(tbl_meta)
        .key_value_metadata(move(user_data))
        .compression(comp_type)
        .stats_level(stat_freq)
        .int96_timestamps(_int96_timestamps)
        .write_v2_headers(header_version == "2.0")
        .dictionary_policy(dict_policy)
        .utc_timestamps(False)
        .build()
    )
    if partitions_info is not None:
        partitions.reserve(len(partitions_info))
        for part in partitions_info:
            partitions.push_back(
                cudf_io_types.partition_info(part[0], part[1])
            )
        args.set_partitions(move(partitions))
    if metadata_file_path is not None:
        if is_list_like(metadata_file_path):
            for path in metadata_file_path:
                c_column_chunks_file_paths.push_back(str.encode(path))
        else:
            c_column_chunks_file_paths.push_back(
                str.encode(metadata_file_path)
            )
        args.set_column_chunks_file_paths(move(c_column_chunks_file_paths))
    if row_group_size_bytes is not None:
        args.set_row_group_size_bytes(row_group_size_bytes)
    if row_group_size_rows is not None:
        args.set_row_group_size_rows(row_group_size_rows)
    if max_page_size_bytes is not None:
        args.set_max_page_size_bytes(max_page_size_bytes)
    if max_page_size_rows is not None:
        args.set_max_page_size_rows(max_page_size_rows)
    if max_dictionary_size is not None:
        args.set_max_dictionary_size(max_dictionary_size)

    with nogil:
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

    Parameters
    ----------
    filepath_or_buffer : str, io.IOBase, os.PathLike, or list
        File path or buffer to write to. The argument may also correspond
        to a list of file paths or buffers.
    index : bool or None, default None
        If ``True``, include a dataframe's index(es) in the file output.
        If ``False``, they will not be written to the file. If ``None``,
        index(es) other than RangeIndex will be saved as columns.
    compression : {'snappy', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    statistics : {'ROWGROUP', 'PAGE', 'COLUMN', 'NONE'}, default 'ROWGROUP'
        Level at which column statistics should be included in file.
    row_group_size_bytes: int, default 134217728
        Maximum size of each stripe of the output.
        By default, 134217728 (128MB) will be used.
    row_group_size_rows: int, default 1000000
        Maximum number of rows of each stripe of the output.
        By default, 1000000 (10^6 rows) will be used.
    max_page_size_bytes: int, default 524288
        Maximum uncompressed size of each page of the output.
        By default, 524288 (512KB) will be used.
    max_page_size_rows: int, default 20000
        Maximum number of rows of each page of the output.
        By default, 20000 will be used.
    max_dictionary_size: int, default 1048576
        Maximum size of the dictionary page for each output column chunk. Dictionary
        encoding for column chunks that exceeds this limit will be disabled.
        By default, 1048576 (1MB) will be used.
    use_dictionary : bool, default True
        If ``True``, enable dictionary encoding for Parquet page data
        subject to ``max_dictionary_size`` constraints.
        If ``False``, disable dictionary encoding for Parquet page data.
    See Also
    --------
    cudf.io.parquet.write_parquet
    """
    cdef bool initialized
    cdef unique_ptr[cpp_parquet_chunked_writer] writer
    cdef table_input_metadata tbl_meta
    cdef cudf_io_types.sink_info sink
    cdef vector[unique_ptr[cudf_io_data_sink.data_sink]] _data_sink
    cdef cudf_io_types.statistics_freq stat_freq
    cdef cudf_io_types.compression_type comp_type
    cdef object index
    cdef size_t row_group_size_bytes
    cdef size_type row_group_size_rows
    cdef size_t max_page_size_bytes
    cdef size_type max_page_size_rows
    cdef size_t max_dictionary_size
    cdef cudf_io_types.dictionary_policy dict_policy

    def __cinit__(self, object filepath_or_buffer, object index=None,
                  object compression="snappy", str statistics="ROWGROUP",
                  int row_group_size_bytes=_ROW_GROUP_SIZE_BYTES_DEFAULT,
                  int row_group_size_rows=1000000,
                  int max_page_size_bytes=524288,
                  int max_page_size_rows=20000,
                  int max_dictionary_size=1048576,
                  bool use_dictionary=True):
        filepaths_or_buffers = (
            list(filepath_or_buffer)
            if is_list_like(filepath_or_buffer)
            else [filepath_or_buffer]
        )
        self.sink = make_sinks_info(filepaths_or_buffers, self._data_sink)
        self.stat_freq = _get_stat_freq(statistics)
        self.comp_type = _get_comp_type(compression)
        self.index = index
        self.initialized = False
        self.row_group_size_bytes = row_group_size_bytes
        self.row_group_size_rows = row_group_size_rows
        self.max_page_size_bytes = max_page_size_bytes
        self.max_page_size_rows = max_page_size_rows
        self.max_dictionary_size = max_dictionary_size
        self.dict_policy = (
            cudf_io_types.dictionary_policy.ADAPTIVE
            if use_dictionary
            else cudf_io_types.dictionary_policy.NEVER
        )

    def write_table(self, table, object partitions_info=None):
        """ Writes a single table to the file """
        if not self.initialized:
            self._initialize_chunked_state(
                table,
                num_partitions=len(partitions_info) if partitions_info else 1
            )

        cdef table_view tv
        if self.index is not False and (
            table._index.name is not None or
                isinstance(table._index, cudf.core.multiindex.MultiIndex)):
            tv = table_view_from_table(table)
        else:
            tv = table_view_from_table(table, ignore_index=True)

        cdef vector[cudf_io_types.partition_info] partitions
        if partitions_info is not None:
            for part in partitions_info:
                partitions.push_back(
                    cudf_io_types.partition_info(part[0], part[1])
                )

        with nogil:
            self.writer.get()[0].write(tv, partitions)

    def close(self, object metadata_file_path=None):
        cdef unique_ptr[vector[uint8_t]] out_metadata_c
        cdef vector[string] column_chunks_file_paths

        if not self.initialized:
            return None

        # Update metadata-collection options
        if metadata_file_path is not None:
            if is_list_like(metadata_file_path):
                for path in metadata_file_path:
                    column_chunks_file_paths.push_back(str.encode(path))
            else:
                column_chunks_file_paths.push_back(
                    str.encode(metadata_file_path)
                )

        with nogil:
            out_metadata_c = move(
                self.writer.get()[0].close(column_chunks_file_paths)
            )

        if metadata_file_path is not None:
            out_metadata_py = BufferArrayFromVector.from_unique_ptr(
                move(out_metadata_c)
            )
            return np.asarray(out_metadata_py)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _initialize_chunked_state(self, table, num_partitions=1):
        """ Prepares all the values required to build the
        chunked_parquet_writer_options and creates a writer"""
        cdef table_view tv

        # Set the table_metadata
        num_index_cols_meta = 0
        self.tbl_meta = table_input_metadata(
            table_view_from_table(table, ignore_index=True))
        if self.index is not False:
            if isinstance(table._index, cudf.core.multiindex.MultiIndex):
                tv = table_view_from_table(table)
                self.tbl_meta = table_input_metadata(tv)
                for level, idx_name in enumerate(table._index.names):
                    self.tbl_meta.column_metadata[level].set_name(
                        (str.encode(idx_name))
                    )
                num_index_cols_meta = len(table._index.names)
            else:
                if table._index.name is not None:
                    tv = table_view_from_table(table)
                    self.tbl_meta = table_input_metadata(tv)
                    self.tbl_meta.column_metadata[0].set_name(
                        str.encode(table._index.name)
                    )
                    num_index_cols_meta = 1

        for i, name in enumerate(table._column_names, num_index_cols_meta):
            self.tbl_meta.column_metadata[i].set_name(name.encode())
            _set_col_metadata(
                table[name]._column,
                self.tbl_meta.column_metadata[i],
            )

        index = (
            False if isinstance(table._index, cudf.RangeIndex) else self.index
        )
        pandas_metadata = generate_pandas_metadata(table, index)
        cdef map[string, string] tmp_user_data
        tmp_user_data[str.encode("pandas")] = str.encode(pandas_metadata)
        cdef vector[map[string, string]] user_data
        user_data = vector[map[string, string]](num_partitions, tmp_user_data)

        cdef chunked_parquet_writer_options args
        with nogil:
            args = move(
                chunked_parquet_writer_options.builder(self.sink)
                .metadata(self.tbl_meta)
                .key_value_metadata(move(user_data))
                .compression(self.comp_type)
                .stats_level(self.stat_freq)
                .row_group_size_bytes(self.row_group_size_bytes)
                .row_group_size_rows(self.row_group_size_rows)
                .max_page_size_bytes(self.max_page_size_bytes)
                .max_page_size_rows(self.max_page_size_rows)
                .max_dictionary_size(self.max_dictionary_size)
                .build()
            )
            args.set_dictionary_policy(self.dict_policy)
            self.writer.reset(new cpp_parquet_chunked_writer(args))
        self.initialized = True


cpdef merge_filemetadata(object filemetadata_list):
    """
    Cython function to call into libcudf API, see `merge_row_group_metadata`.

    See Also
    --------
    cudf.io.parquet.merge_row_group_metadata
    """
    cdef vector[unique_ptr[vector[uint8_t]]] list_c
    cdef vector[uint8_t] blob_c
    cdef unique_ptr[vector[uint8_t]] output_c

    for blob_py in filemetadata_list:
        blob_c = blob_py
        list_c.push_back(move(make_unique[vector[uint8_t]](blob_c)))

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
    elif statistics == "COLUMN":
        return cudf_io_types.statistics_freq.STATISTICS_COLUMN
    else:
        raise ValueError("Unsupported `statistics_freq` type")


cdef cudf_io_types.compression_type _get_comp_type(object compression):
    if compression is None:
        return cudf_io_types.compression_type.NONE

    compression = str(compression).upper()
    if compression == "SNAPPY":
        return cudf_io_types.compression_type.SNAPPY
    elif compression == "ZSTD":
        return cudf_io_types.compression_type.ZSTD
    elif compression == "LZ4":
        return cudf_io_types.compression_type.LZ4
    else:
        raise ValueError("Unsupported `compression` type")


cdef _set_col_metadata(
    Column col,
    column_in_metadata& col_meta,
    bool force_nullable_schema=False,
):
    if force_nullable_schema:
        # Only set nullability if `force_nullable_schema`
        # is true.
        col_meta.set_nullability(True)

    if isinstance(col.dtype, cudf.StructDtype):
        for i, (child_col, name) in enumerate(
            zip(col.children, list(col.dtype.fields))
        ):
            col_meta.child(i).set_name(name.encode())
            _set_col_metadata(
                child_col,
                col_meta.child(i),
                force_nullable_schema
            )
    elif isinstance(col.dtype, cudf.ListDtype):
        _set_col_metadata(
            col.children[1],
            col_meta.child(1),
            force_nullable_schema
        )
    elif isinstance(col.dtype, cudf.core.dtypes.DecimalDtype):
        col_meta.set_decimal_precision(col.dtype.precision)
