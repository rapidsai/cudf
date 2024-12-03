# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int64_t
from libcpp cimport bool, int
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
import itertools
from collections import OrderedDict

try:
    import ujson as json
except ImportError:
    import json

cimport pylibcudf.libcudf.lists.lists_column_view as cpp_lists_column_view

from cudf._lib.column cimport Column
from cudf._lib.io.utils cimport update_col_struct_field_names
from cudf._lib.utils cimport data_from_pylibcudf_io

import pylibcudf as plc

import cudf
from cudf._lib.types import SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES
from cudf._lib.utils import _index_level_name, generate_pandas_metadata
from cudf.core.buffer import acquire_spill_lock
from pylibcudf.io.types cimport TableInputMetadata, SinkInfo, ColumnInMetadata
from pylibcudf.io.orc cimport OrcChunkedWriter

# TODO: Consider inlining this function since it seems to only be used in one place.
cpdef read_parsed_orc_statistics(filepath_or_buffer):
    """
    Cython function to call into libcudf API, see `read_parsed_orc_statistics`.

    See Also
    --------
    cudf.io.orc.read_orc_statistics
    """

    parsed = (
        plc.io.orc.read_parsed_orc_statistics(
            plc.io.SourceInfo([filepath_or_buffer])
        )
    )

    return parsed.column_names, parsed.file_stats, parsed.stripes_stats


cpdef read_orc(object filepaths_or_buffers,
               object columns=None,
               object stripes=None,
               object skip_rows=None,
               object num_rows=None,
               bool use_index=True,
               object timestamp_type=None):
    """
    Cython function to call into libcudf API, see `read_orc`.

    See Also
    --------
    cudf.read_orc

    Notes
    -----
    Currently this function only considers the metadata of the first file in the list of
    filepaths_or_buffers.
    """

    if columns is not None:
        columns = [str(col) for col in columns]

    tbl_w_meta = plc.io.orc.read_orc(
        plc.io.SourceInfo(filepaths_or_buffers),
        columns,
        stripes,
        get_skiprows_arg(skip_rows),
        get_num_rows_arg(num_rows),
        use_index,
        plc.types.DataType(
            SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES[
                cudf.dtype(timestamp_type)
            ]
        )
    )

    names = tbl_w_meta.column_names(include_children=False)

    actual_index_names, col_names, is_range_index, reset_index_name, \
        range_idx = _get_index_from_metadata(tbl_w_meta.per_file_user_data,
                                             names,
                                             skip_rows,
                                             num_rows)

    if columns is not None and (isinstance(columns, list) and len(columns) == 0):
        # When `columns=[]`, index needs to be
        # established, but not the columns.
        nrows = tbl_w_meta.tbl.num_rows()
        return {}, cudf.RangeIndex(nrows)

    data, index = data_from_pylibcudf_io(
        tbl_w_meta,
        col_names if columns is None else names,
        actual_index_names
    )

    if is_range_index:
        index = range_idx
    elif reset_index_name:
        index.names = [None] * len(index.names)

    child_name_values = tbl_w_meta.child_names.values()

    data = {
        name: update_col_struct_field_names(
            col, child_names
        )
        for (name, col), child_names in zip(data.items(), child_name_values)
    }

    return data, index


def _get_comp_type(object compression):
    if compression is None or compression is False:
        return plc.io.types.CompressionType.NONE

    compression = str(compression).upper()
    if compression == "SNAPPY":
        return plc.io.types.CompressionType.SNAPPY
    elif compression == "ZLIB":
        return plc.io.types.CompressionType.ZLIB
    elif compression == "ZSTD":
        return plc.io.types.CompressionType.ZSTD
    elif compression == "LZ4":
        return plc.io.types.CompressionType.LZ4
    else:
        raise ValueError(f"Unsupported `compression` type {compression}")


cdef tuple _get_index_from_metadata(
        vector[map[string, string]] user_data,
        object names,
        object skip_rows,
        object num_rows):

    meta = None
    index_col = None
    is_range_index = False
    reset_index_name = False
    range_idx = None

    if user_data.size() > 0:
        json_str = user_data[0][b'pandas'].decode('utf-8')
        if json_str != "":
            meta = json.loads(json_str)
            if 'index_columns' in meta and len(meta['index_columns']) > 0:
                index_col = meta['index_columns']
                if isinstance(index_col[0], dict) and \
                        index_col[0]['kind'] == 'range':
                    is_range_index = True
                else:
                    index_col_names = OrderedDict()
                    for idx_col in index_col:
                        for c in meta['columns']:
                            if c['field_name'] == idx_col:
                                index_col_names[idx_col] = \
                                    c['name'] or c['field_name']
                                if c['name'] is None:
                                    reset_index_name = True

    actual_index_names = None
    if index_col is not None and len(index_col) > 0:
        if is_range_index:
            range_index_meta = index_col[0]
            range_idx = cudf.RangeIndex(
                start=range_index_meta['start'],
                stop=range_index_meta['stop'],
                step=range_index_meta['step'],
                name=range_index_meta['name']
            )
            if skip_rows is not None:
                range_idx = range_idx[skip_rows:]
            if num_rows is not None:
                range_idx = range_idx[:num_rows]
        else:
            actual_index_names = list(index_col_names.values())
            names = names[len(actual_index_names):]

    return (
        actual_index_names,
        names,
        is_range_index,
        reset_index_name,
        range_idx
    )


def _get_orc_stat_freq(str statistics):
    """
    Convert ORC statistics terms to CUDF convention:
      - ORC "STRIPE"   == CUDF "ROWGROUP"
      - ORC "ROWGROUP" == CUDF "PAGE"
    """
    statistics = str(statistics).upper()
    if statistics == "NONE":
        return plc.io.types.StatisticsFreq.STATISTICS_NONE
    elif statistics == "STRIPE":
        return plc.io.types.StatisticsFreq.STATISTICS_ROWGROUP
    elif statistics == "ROWGROUP":
        return plc.io.types.StatisticsFreq.STATISTICS_PAGE
    else:
        raise ValueError(f"Unsupported `statistics_freq` type {statistics}")


@acquire_spill_lock()
def write_orc(
    table,
    object path_or_buf,
    object compression="snappy",
    str statistics="ROWGROUP",
    object stripe_size_bytes=None,
    object stripe_size_rows=None,
    object row_index_stride=None,
    object cols_as_map_type=None,
    object index=None
):
    """
    Cython function to call into libcudf API, see `cudf::io::write_orc`.

    See Also
    --------
    cudf.read_orc
    """
    user_data = {}
    user_data["pandas"] = generate_pandas_metadata(table, index)
    if index is True or (
        index is None and not isinstance(table._index, cudf.RangeIndex)
    ):
        columns = table._columns if table._index is None else [
            *table.index._columns, *table._columns
        ]
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

    if cols_as_map_type is not None:
        cols_as_map_type = set(cols_as_map_type)

    for i, name in enumerate(table._column_names, num_index_cols_meta):
        tbl_meta.column_metadata[i].set_name(name)
        _set_col_children_metadata(
            table[name]._column,
            tbl_meta.column_metadata[i],
            (cols_as_map_type is not None)
            and (name in cols_as_map_type),
        )

    options = (
        plc.io.orc.OrcWriterOptions.builder(
            plc.io.SinkInfo([path_or_buf]), plc_table
        )
        .metadata(tbl_meta)
        .key_value_metadata(user_data)
        .compression(_get_comp_type(compression))
        .enable_statistics(_get_orc_stat_freq(statistics))
        .build()
    )
    if stripe_size_bytes is not None:
        options.set_stripe_size_bytes(stripe_size_bytes)
    if stripe_size_rows is not None:
        options.set_stripe_size_rows(stripe_size_rows)
    if row_index_stride is not None:
        options.set_row_index_stride(row_index_stride)

    plc.io.orc.write_orc(options)


cdef int64_t get_skiprows_arg(object arg) except*:
    arg = 0 if arg is None else arg
    if not isinstance(arg, int) or arg < 0:
        raise TypeError("skiprows must be an int >= 0")
    return <int64_t> arg

cdef int64_t get_num_rows_arg(object arg) except*:
    arg = -1 if arg is None else arg
    if not isinstance(arg, int) or arg < -1:
        raise TypeError("num_rows must be an int >= -1")
    return <int64_t> arg


cdef class ORCWriter:
    """
    ORCWriter lets you you incrementally write out a ORC file from a series
    of cudf tables

    See Also
    --------
    cudf.io.orc.to_orc
    """
    cdef bool initialized
    cdef OrcChunkedWriter writer
    cdef SinkInfo sink
    cdef str statistics
    cdef object compression
    cdef object index
    cdef TableInputMetadata tbl_meta
    cdef object cols_as_map_type
    cdef object stripe_size_bytes
    cdef object stripe_size_rows
    cdef object row_index_stride

    def __cinit__(self,
                  object path,
                  object index=None,
                  object compression="snappy",
                  str statistics="ROWGROUP",
                  object cols_as_map_type=None,
                  object stripe_size_bytes=None,
                  object stripe_size_rows=None,
                  object row_index_stride=None):
        self.sink = plc.io.SinkInfo([path])
        self.statistics = statistics
        self.compression = compression
        self.index = index
        self.cols_as_map_type = cols_as_map_type \
            if cols_as_map_type is None else set(cols_as_map_type)
        self.stripe_size_bytes = stripe_size_bytes
        self.stripe_size_rows = stripe_size_rows
        self.row_index_stride = row_index_stride
        self.initialized = False

    def write_table(self, table):
        """ Writes a single table to the file """
        if not self.initialized:
            self._initialize_chunked_state(table)

        keep_index = self.index is not False and (
            table._index.name is not None or
            isinstance(table._index, cudf.core.multiindex.MultiIndex)
        )
        if keep_index:
            columns = [
                col.to_pylibcudf(mode="read")
                for col in itertools.chain(table.index._columns, table._columns)
            ]
        else:
            columns = [col.to_pylibcudf(mode="read") for col in table._columns]

        self.writer.write(plc.Table(columns))

    def close(self):
        if not self.initialized:
            return

        self.writer.close()

    def __dealloc__(self):
        self.close()

    def _initialize_chunked_state(self, table):
        """
        Prepare all the values required to build the
        chunked_orc_writer_options anb creates a writer"""

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
                    self.tbl_meta.column_metadata[level].set_name(
                        idx_name
                    )
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
                    self.tbl_meta.column_metadata[0].set_name(
                        table._index.name
                    )
                    num_index_cols_meta = 1

        for i, name in enumerate(table._column_names, num_index_cols_meta):
            self.tbl_meta.column_metadata[i].set_name(name)
            _set_col_children_metadata(
                table[name]._column,
                self.tbl_meta.column_metadata[i],
                (self.cols_as_map_type is not None)
                and (name in self.cols_as_map_type),
            )

        user_data = {}
        pandas_metadata = generate_pandas_metadata(table, self.index)
        user_data["pandas"] = pandas_metadata

        options = (
            plc.io.orc.ChunkedOrcWriterOptions.builder(self.sink)
            .metadata(self.tbl_meta)
            .key_value_metadata(user_data)
            .compression(_get_comp_type(self.compression))
            .enable_statistics(_get_orc_stat_freq(self.statistics))
            .build()
        )
        if self.stripe_size_bytes is not None:
            options.set_stripe_size_bytes(self.stripe_size_bytes)
        if self.stripe_size_rows is not None:
            options.set_stripe_size_rows(self.stripe_size_rows)
        if self.row_index_stride is not None:
            options.set_row_index_stride(self.row_index_stride)

        self.writer = plc.io.orc.OrcChunkedWriter.from_options(options)

        self.initialized = True

cdef _set_col_children_metadata(Column col,
                                ColumnInMetadata col_meta,
                                list_column_as_map=False):
    if isinstance(col.dtype, cudf.StructDtype):
        for i, (child_col, name) in enumerate(
            zip(col.children, list(col.dtype.fields))
        ):
            col_meta.child(i).set_name(name)
            _set_col_children_metadata(
                child_col, col_meta.child(i), list_column_as_map
            )
    elif isinstance(col.dtype, cudf.ListDtype):
        if list_column_as_map:
            col_meta.set_list_column_as_map()
        _set_col_children_metadata(
            col.children[cpp_lists_column_view.child_column_index],
            col_meta.child(cpp_lists_column_view.child_column_index),
            list_column_as_map
        )
    else:
        return
