# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import io

import pyarrow as pa
import itertools
import cudf
from cudf.core.buffer import acquire_spill_lock

try:
    import ujson as json
except ImportError:
    import json

import numpy as np

from cudf.api.types import is_list_like

from cudf._lib.utils cimport _data_from_columns, data_from_pylibcudf_io

from cudf._lib.utils import _index_level_name, generate_pandas_metadata

from libc.stdint cimport int64_t
from libcpp cimport bool

from pylibcudf.expressions cimport Expression
from pylibcudf.io.parquet cimport ChunkedParquetReader
from pylibcudf.libcudf.io.types cimport (
    statistics_freq,
    compression_type,
    dictionary_policy,
)
from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column
from cudf._lib.io.utils cimport (
    add_df_col_struct_names,
)

import pylibcudf as plc

from pylibcudf cimport Table

from cudf.utils.ioutils import _ROW_GROUP_SIZE_BYTES_DEFAULT
from pylibcudf.io.types cimport TableInputMetadata, SinkInfo, ColumnInMetadata
from pylibcudf.io.parquet cimport ParquetChunkedWriter


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


cdef object _process_metadata(object df,
                              list names,
                              dict child_names,
                              list per_file_user_data,
                              object row_groups,
                              object filepaths_or_buffers,
                              bool allow_range_index,
                              bool use_pandas_metadata,
                              size_type nrows=-1,
                              int64_t skip_rows=0,
                              ):

    add_df_col_struct_names(df, child_names)
    index_col = None
    is_range_index = True
    column_index_type = None
    index_col_names = None
    meta = None
    for single_file in per_file_user_data:
        if b'pandas' not in single_file:
            continue
        json_str = single_file[b'pandas'].decode('utf-8')
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
                    ) for s in filepaths_or_buffers
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
                    idx = cudf.Index._from_column(cudf.core.column.column_empty(0))
            else:
                start = range_index_meta["start"] + skip_rows
                stop = range_index_meta["stop"]
                if nrows > -1:
                    stop = start + nrows
                idx = cudf.RangeIndex(
                    start=start,
                    stop=stop,
                    step=range_index_meta['step'],
                    name=range_index_meta['name']
                )

            df._index = idx
        elif set(index_col).issubset(names):
            index_data = df[index_col]
            actual_index_names = iter(index_col_names.values())
            if index_data._num_columns == 1:
                idx = cudf.Index._from_column(
                    index_data._columns[0],
                    name=next(actual_index_names)
                )
            else:
                idx = cudf.MultiIndex.from_frame(
                    index_data,
                    names=list(actual_index_names)
                )
            df.drop(columns=index_col, inplace=True)
            df._index = idx
        else:
            if use_pandas_metadata:
                df.index.names = index_col

    if df._num_columns == 0 and column_index_type is not None:
        df._data.label_dtype = cudf.dtype(column_index_type)

    return df


def read_parquet_chunked(
    filepaths_or_buffers,
    columns=None,
    row_groups=None,
    use_pandas_metadata=True,
    size_t chunk_read_limit=0,
    size_t pass_read_limit=1024000000,
    size_type nrows=-1,
    int64_t skip_rows=0,
    allow_mismatched_pq_schemas=False
):
    # Note: If this function ever takes accepts filters
    # allow_range_index needs to be False when a filter is passed
    # (see read_parquet)
    allow_range_index = columns is not None and len(columns) != 0

    options = (
        plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(filepaths_or_buffers)
        )
        .use_pandas_metadata(use_pandas_metadata)
        .allow_mismatched_pq_schemas(allow_mismatched_pq_schemas)
        .build()
    )
    if row_groups is not None:
        options.set_row_groups(row_groups)
    if nrows > -1:
        options.set_num_rows(nrows)
    if skip_rows != 0:
        options.set_skip_rows(skip_rows)
    if columns is not None:
        options.set_columns(columns)

    reader = ChunkedParquetReader(
        options,
        chunk_read_limit=chunk_read_limit,
        pass_read_limit=pass_read_limit,
    )

    tbl_w_meta = reader.read_chunk()
    column_names = tbl_w_meta.column_names(include_children=False)
    child_names = tbl_w_meta.child_names
    per_file_user_data = tbl_w_meta.per_file_user_data
    concatenated_columns = tbl_w_meta.tbl.columns()

    # save memory
    del tbl_w_meta

    cdef Table tbl
    while reader.has_next():
        tbl = reader.read_chunk().tbl

        for i in range(tbl.num_columns()):
            concatenated_columns[i] = plc.concatenate.concatenate(
                [concatenated_columns[i], tbl._columns[i]]
            )
            # Drop residual columns to save memory
            tbl._columns[i] = None

    df = cudf.DataFrame._from_data(
        *_data_from_columns(
            columns=[Column.from_pylibcudf(plc) for plc in concatenated_columns],
            column_names=column_names,
            index_names=None
        )
    )
    df = _process_metadata(df, column_names, child_names,
                           per_file_user_data, row_groups,
                           filepaths_or_buffers,
                           allow_range_index, use_pandas_metadata,
                           nrows=nrows, skip_rows=skip_rows)
    return df


cpdef read_parquet(filepaths_or_buffers, columns=None, row_groups=None,
                   use_pandas_metadata=True,
                   Expression filters=None,
                   size_type nrows=-1,
                   int64_t skip_rows=0,
                   allow_mismatched_pq_schemas=False):
    """
    Cython function to call into libcudf API, see `read_parquet`.

    filters, if not None, should be an Expression that evaluates to a
    boolean predicate as a function of columns being read.

    See Also
    --------
    cudf.io.parquet.read_parquet
    cudf.io.parquet.to_parquet
    """

    allow_range_index = True
    if columns is not None and len(columns) == 0 or filters:
        allow_range_index = False

    options = (
        plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(filepaths_or_buffers)
        )
        .use_pandas_metadata(use_pandas_metadata)
        .allow_mismatched_pq_schemas(allow_mismatched_pq_schemas)
        .build()
    )
    if row_groups is not None:
        options.set_row_groups(row_groups)
    if nrows > -1:
        options.set_num_rows(nrows)
    if skip_rows != 0:
        options.set_skip_rows(skip_rows)
    if columns is not None:
        options.set_columns(columns)
    if filters is not None:
        options.set_filter(filters)

    tbl_w_meta = plc.io.parquet.read_parquet(options)

    df = cudf.DataFrame._from_data(
        *data_from_pylibcudf_io(tbl_w_meta)
    )

    df = _process_metadata(df, tbl_w_meta.column_names(include_children=False),
                           tbl_w_meta.child_names, tbl_w_meta.per_file_user_data,
                           row_groups, filepaths_or_buffers,
                           allow_range_index, use_pandas_metadata,
                           nrows=nrows, skip_rows=skip_rows)
    return df

cpdef read_parquet_metadata(list filepaths_or_buffers):
    """
    Cython function to call into libcudf API, see `read_parquet_metadata`.

    See Also
    --------
    cudf.io.parquet.read_parquet
    cudf.io.parquet.to_parquet
    """
    parquet_metadata = plc.io.parquet_metadata.read_parquet_metadata(
        plc.io.SourceInfo(filepaths_or_buffers)
    )

    # read all column names including index column, if any
    col_names = [info.name() for info in parquet_metadata.schema().root().children()]

    index_col_names = set()
    json_str = parquet_metadata.metadata()['pandas']
    if json_str != "":
        meta = json.loads(json_str)
        file_is_range_index, index_col, _ = _parse_metadata(meta)
        if (
            not file_is_range_index
            and index_col is not None
        ):
            columns = meta['columns']
            for idx_col in index_col:
                for c in columns:
                    if c['field_name'] == idx_col:
                        index_col_names.add(idx_col)

    # remove the index column from the list of column names
    # only if index_col_names is not None
    if len(index_col_names) >= 0:
        col_names = [name for name in col_names if name not in index_col_names]

    return (
        parquet_metadata.num_rows(),
        parquet_metadata.num_rowgroups(),
        col_names,
        len(col_names),
        parquet_metadata.rowgroup_metadata()
    )


@acquire_spill_lock()
def write_parquet(
    table,
    object filepaths_or_buffers,
    object index=None,
    object compression="snappy",
    object statistics="ROWGROUP",
    object metadata_file_path=None,
    object int96_timestamps=False,
    object row_group_size_bytes=None,
    object row_group_size_rows=None,
    object max_page_size_bytes=None,
    object max_page_size_rows=None,
    object max_dictionary_size=None,
    object partitions_info=None,
    object force_nullable_schema=False,
    header_version="1.0",
    use_dictionary=True,
    object skip_compression=None,
    object column_encoding=None,
    object column_type_length=None,
    object output_as_binary=None,
    write_arrow_schema=False,
):
    """
    Cython function to call into libcudf API, see `write_parquet`.

    See Also
    --------
    cudf.io.parquet.write_parquet
    """
    if index is True or (
        index is None and not isinstance(table._index, cudf.RangeIndex)
    ):
        columns = [*table.index._columns, *table._columns]
        plc_table = plc.Table([col.to_pylibcudf(mode="read") for col in columns])
        tbl_meta = TableInputMetadata(plc_table)
        for level, idx_name in enumerate(table._index.names):
            tbl_meta.column_metadata[level].set_name(
                _index_level_name(idx_name, level, table._column_names)
            )
        num_index_cols_meta = len(table._index.names)
    else:
        plc_table = plc.Table(
            [col.to_pylibcudf(mode="read") for col in table._columns]
        )
        tbl_meta = TableInputMetadata(plc_table)
        num_index_cols_meta = 0

    for i, name in enumerate(table._column_names, num_index_cols_meta):
        if not isinstance(name, str):
            if cudf.get_option("mode.pandas_compatible"):
                tbl_meta.column_metadata[i].set_name(str(name))
            else:
                raise ValueError(
                    "Writing a Parquet file requires string column names"
                )
        else:
            tbl_meta.column_metadata[i].set_name(name)

        _set_col_metadata(
            table[name]._column,
            tbl_meta.column_metadata[i],
            force_nullable_schema,
            None,
            skip_compression,
            column_encoding,
            column_type_length,
            output_as_binary
        )
    if partitions_info is not None:
        user_data = [
            {"pandas": generate_pandas_metadata(
                table.iloc[start_row:start_row + num_row].copy(deep=False),
                index
            )}
            for start_row, num_row in partitions_info
        ]
    else:
        user_data = [{"pandas": generate_pandas_metadata(table, index)}]

    if header_version not in ("1.0", "2.0"):
        raise ValueError(
            f"Invalid parquet header version: {header_version}. "
            "Valid values are '1.0' and '2.0'"
        )

    dict_policy = (
        plc.io.types.DictionaryPolicy.ADAPTIVE
        if use_dictionary
        else plc.io.types.DictionaryPolicy.NEVER
    )

    comp_type = _get_comp_type(compression)
    stat_freq = _get_stat_freq(statistics)
    options = (
        plc.io.parquet.ParquetWriterOptions.builder(
            plc.io.SinkInfo(filepaths_or_buffers), plc_table
        )
        .metadata(tbl_meta)
        .key_value_metadata(user_data)
        .compression(comp_type)
        .stats_level(stat_freq)
        .int96_timestamps(int96_timestamps)
        .write_v2_headers(header_version == "2.0")
        .dictionary_policy(dict_policy)
        .utc_timestamps(False)
        .write_arrow_schema(write_arrow_schema)
        .build()
    )
    if partitions_info is not None:
        options.set_partitions(
            [plc.io.types.PartitionInfo(part[0], part[1]) for part in partitions_info]
        )
    if metadata_file_path is not None:
        if is_list_like(metadata_file_path):
            options.set_column_chunks_file_paths(metadata_file_path)
        else:
            options.set_column_chunks_file_paths([metadata_file_path])
    if row_group_size_bytes is not None:
        options.set_row_group_size_bytes(row_group_size_bytes)
    if row_group_size_rows is not None:
        options.set_row_group_size_rows(row_group_size_rows)
    if max_page_size_bytes is not None:
        options.set_max_page_size_bytes(max_page_size_bytes)
    if max_page_size_rows is not None:
        options.set_max_page_size_rows(max_page_size_rows)
    if max_dictionary_size is not None:
        options.set_max_dictionary_size(max_dictionary_size)
    blob = plc.io.parquet.write_parquet(options)
    if metadata_file_path is not None:
        return np.asarray(blob.obj)
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
    row_group_size_bytes: int, default ``uint64 max``
        Maximum size of each stripe of the output.
        By default, a virtually infinite size equal to ``uint64 max`` will be used.
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
    store_schema : bool, default False
        If ``True``, enable computing and writing arrow schema to Parquet
        file footer's key-value metadata section for faithful round-tripping.
    See Also
    --------
    cudf.io.parquet.write_parquet
    """
    cdef bool initialized
    cdef ParquetChunkedWriter writer
    cdef SinkInfo sink
    cdef TableInputMetadata tbl_meta
    cdef str statistics
    cdef object compression
    cdef object index
    cdef size_t row_group_size_bytes
    cdef size_type row_group_size_rows
    cdef size_t max_page_size_bytes
    cdef size_type max_page_size_rows
    cdef size_t max_dictionary_size
    cdef bool use_dictionary
    cdef bool write_arrow_schema

    def __cinit__(self, object filepath_or_buffer, object index=None,
                  object compression="snappy", str statistics="ROWGROUP",
                  size_t row_group_size_bytes=_ROW_GROUP_SIZE_BYTES_DEFAULT,
                  size_type row_group_size_rows=1000000,
                  size_t max_page_size_bytes=524288,
                  size_type max_page_size_rows=20000,
                  size_t max_dictionary_size=1048576,
                  bool use_dictionary=True,
                  bool store_schema=False):
        filepaths_or_buffers = (
            list(filepath_or_buffer)
            if is_list_like(filepath_or_buffer)
            else [filepath_or_buffer]
        )
        self.sink = plc.io.SinkInfo(filepaths_or_buffers)
        self.statistics = statistics
        self.compression = compression
        self.index = index
        self.initialized = False
        self.row_group_size_bytes = row_group_size_bytes
        self.row_group_size_rows = row_group_size_rows
        self.max_page_size_bytes = max_page_size_bytes
        self.max_page_size_rows = max_page_size_rows
        self.max_dictionary_size = max_dictionary_size
        self.use_dictionary = use_dictionary
        self.write_arrow_schema = store_schema

    def write_table(self, table, object partitions_info=None):
        """ Writes a single table to the file """
        if not self.initialized:
            self._initialize_chunked_state(
                table,
                num_partitions=len(partitions_info) if partitions_info else 1
            )
        if self.index is not False and (
            table._index.name is not None or
                isinstance(table._index, cudf.core.multiindex.MultiIndex)):
            columns = [*table.index._columns, *table._columns]
            plc_table = plc.Table([col.to_pylibcudf(mode="read") for col in columns])
        else:
            plc_table = plc.Table(
                [col.to_pylibcudf(mode="read") for col in table._columns]
            )
        self.writer.write(plc_table, partitions_info)

    def close(self, object metadata_file_path=None):
        if not self.initialized:
            return None
        column_chunks_file_paths=[]
        if metadata_file_path is not None:
            if is_list_like(metadata_file_path):
                column_chunks_file_paths = list(metadata_file_path)
            else:
                column_chunks_file_paths = [metadata_file_path]
        blob = self.writer.close(column_chunks_file_paths)
        if metadata_file_path is not None:
            return np.asarray(blob.obj)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _initialize_chunked_state(self, table, num_partitions=1):
        """ Prepares all the values required to build the
        chunked_parquet_writer_options and creates a writer"""

        # Set the table_metadata
        num_index_cols_meta = 0
        plc_table = plc.Table(
            [
                col.to_pylibcudf(mode="read")
                for col in table._columns
            ]
        )
        self.tbl_meta = TableInputMetadata(plc_table)
        if self.index is not False:
            if isinstance(table._index, cudf.core.multiindex.MultiIndex):
                plc_table = plc.Table(
                    [
                        col.to_pylibcudf(mode="read")
                        for col in itertools.chain(table.index._columns, table._columns)
                    ]
                )
                self.tbl_meta = TableInputMetadata(plc_table)
                for level, idx_name in enumerate(table._index.names):
                    self.tbl_meta.column_metadata[level].set_name(idx_name)
                num_index_cols_meta = len(table._index.names)
            else:
                if table._index.name is not None:
                    plc_table = plc.Table(
                        [
                            col.to_pylibcudf(mode="read")
                            for col in itertools.chain(
                                table.index._columns, table._columns
                            )
                        ]
                    )
                    self.tbl_meta = TableInputMetadata(plc_table)
                    self.tbl_meta.column_metadata[0].set_name(table._index.name)
                    num_index_cols_meta = 1

        for i, name in enumerate(table._column_names, num_index_cols_meta):
            self.tbl_meta.column_metadata[i].set_name(name)
            _set_col_metadata(
                table[name]._column,
                self.tbl_meta.column_metadata[i],
            )

        index = (
            False if isinstance(table._index, cudf.RangeIndex) else self.index
        )
        user_data = [{"pandas" : generate_pandas_metadata(table, index)}]*num_partitions
        cdef compression_type comp_type = _get_comp_type(self.compression)
        cdef statistics_freq stat_freq = _get_stat_freq(self.statistics)
        cdef dictionary_policy dict_policy = (
            plc.io.types.DictionaryPolicy.ADAPTIVE
            if self.use_dictionary
            else plc.io.types.DictionaryPolicy.NEVER
        )
        options = (
            plc.io.parquet.ChunkedParquetWriterOptions.builder(self.sink)
            .metadata(self.tbl_meta)
            .key_value_metadata(user_data)
            .compression(comp_type)
            .stats_level(stat_freq)
            .row_group_size_bytes(self.row_group_size_bytes)
            .row_group_size_rows(self.row_group_size_rows)
            .max_page_size_bytes(self.max_page_size_bytes)
            .max_page_size_rows(self.max_page_size_rows)
            .max_dictionary_size(self.max_dictionary_size)
            .write_arrow_schema(self.write_arrow_schema)
            .build()
        )
        options.set_dictionary_policy(dict_policy)
        self.writer = plc.io.parquet.ParquetChunkedWriter.from_options(options)
        self.initialized = True


cpdef merge_filemetadata(object filemetadata_list):
    """
    Cython function to call into libcudf API, see `merge_row_group_metadata`.

    See Also
    --------
    cudf.io.parquet.merge_row_group_metadata
    """
    return np.asarray(
        plc.io.parquet.merge_row_group_metadata(filemetadata_list).obj
    )


cdef statistics_freq _get_stat_freq(str statistics):
    result = getattr(
        plc.io.types.StatisticsFreq,
        f"STATISTICS_{statistics.upper()}",
        None
    )
    if result is None:
        raise ValueError("Unsupported `statistics_freq` type")
    return result


cdef compression_type _get_comp_type(object compression):
    if compression is None:
        return plc.io.types.CompressionType.NONE
    result = getattr(
        plc.io.types.CompressionType,
        str(compression).upper(),
        None
    )
    if result is None:
        raise ValueError("Unsupported `compression` type")
    return result


cdef _set_col_metadata(
    Column col,
    ColumnInMetadata col_meta,
    bool force_nullable_schema=False,
    str path=None,
    object skip_compression=None,
    object column_encoding=None,
    object column_type_length=None,
    object output_as_binary=None,
):
    need_path = (skip_compression is not None or column_encoding is not None or
                 column_type_length is not None or output_as_binary is not None)
    name = col_meta.get_name() if need_path else None
    full_path = path + "." + name if path is not None else name

    if force_nullable_schema:
        # Only set nullability if `force_nullable_schema`
        # is true.
        col_meta.set_nullability(True)

    if skip_compression is not None and full_path in skip_compression:
        col_meta.set_skip_compression(True)

    if column_encoding is not None and full_path in column_encoding:
        encoding = column_encoding[full_path]
        if encoding is None:
            c_encoding = plc.io.types.ColumnEncoding.USE_DEFAULT
        else:
            enc = str(encoding).upper()
            c_encoding = getattr(plc.io.types.ColumnEncoding, enc, None)
            if c_encoding is None:
                raise ValueError("Unsupported `column_encoding` type")
        col_meta.set_encoding(c_encoding)

    if column_type_length is not None and full_path in column_type_length:
        col_meta.set_output_as_binary(True)
        col_meta.set_type_length(column_type_length[full_path])

    if output_as_binary is not None and full_path in output_as_binary:
        col_meta.set_output_as_binary(True)

    if isinstance(col.dtype, cudf.StructDtype):
        for i, (child_col, name) in enumerate(
            zip(col.children, list(col.dtype.fields))
        ):
            col_meta.child(i).set_name(name)
            _set_col_metadata(
                child_col,
                col_meta.child(i),
                force_nullable_schema,
                full_path,
                skip_compression,
                column_encoding,
                column_type_length,
                output_as_binary
            )
    elif isinstance(col.dtype, cudf.ListDtype):
        if full_path is not None:
            full_path = full_path + ".list"
            col_meta.child(1).set_name("element")
        _set_col_metadata(
            col.children[1],
            col_meta.child(1),
            force_nullable_schema,
            full_path,
            skip_compression,
            column_encoding,
            column_type_length,
            output_as_binary
        )
    elif isinstance(col.dtype, cudf.core.dtypes.DecimalDtype):
        col_meta.set_decimal_precision(col.dtype.precision)
