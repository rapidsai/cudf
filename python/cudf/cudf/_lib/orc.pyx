# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import cudf
from cudf.core.buffer import acquire_spill_lock

from libc.stdint cimport int64_t
from libcpp cimport bool, int
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

import datetime
from collections import OrderedDict

cimport cudf._lib.pylibcudf.libcudf.lists.lists_column_view as cpp_lists_column_view

try:
    import ujson as json
except ImportError:
    import json

cimport cudf._lib.pylibcudf.libcudf.io.types as cudf_io_types
from cudf._lib.column cimport Column
from cudf._lib.io.datasource cimport NativeFileDatasource
from cudf._lib.io.utils cimport (
    make_sink_info,
    make_source_info,
    update_column_struct_field_names,
)
from cudf._lib.pylibcudf.libcudf.io.data_sink cimport data_sink
from cudf._lib.pylibcudf.libcudf.io.orc cimport (
    chunked_orc_writer_options,
    orc_chunked_writer,
    orc_reader_options,
    orc_writer_options,
    read_orc as libcudf_read_orc,
    write_orc as libcudf_write_orc,
)
from cudf._lib.pylibcudf.libcudf.io.orc_metadata cimport (
    binary_statistics,
    bucket_statistics,
    column_statistics,
    date_statistics,
    decimal_statistics,
    double_statistics,
    integer_statistics,
    no_statistics,
    parsed_orc_statistics,
    read_parsed_orc_statistics as libcudf_read_parsed_orc_statistics,
    statistics_type,
    string_statistics,
    timestamp_statistics,
)
from cudf._lib.pylibcudf.libcudf.io.types cimport (
    column_in_metadata,
    compression_type,
    sink_info,
    source_info,
    table_input_metadata,
    table_with_metadata,
)
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport data_type, size_type, type_id
from cudf._lib.variant cimport get_if as std_get_if, holds_alternative

from cudf._lib.types import SUPPORTED_NUMPY_TO_LIBCUDF_TYPES

from cudf._lib.types cimport underlying_type_t_type_id
from cudf._lib.utils cimport data_from_unique_ptr, table_view_from_table

from pyarrow.lib import NativeFile

from cudf._lib.utils import _index_level_name, generate_pandas_metadata


cdef _parse_column_type_statistics(column_statistics stats):
    # Initialize stats to return and parse stats blob
    column_stats = {}

    if stats.number_of_values.has_value():
        column_stats["number_of_values"] = stats.number_of_values.value()

    if stats.has_null.has_value():
        column_stats["has_null"] = stats.has_null.value()

    cdef statistics_type type_specific_stats = stats.type_specific_stats

    cdef integer_statistics* int_stats
    cdef double_statistics* dbl_stats
    cdef string_statistics* str_stats
    cdef bucket_statistics* bucket_stats
    cdef decimal_statistics* dec_stats
    cdef date_statistics* date_stats
    cdef binary_statistics* bin_stats
    cdef timestamp_statistics* ts_stats

    if holds_alternative[no_statistics](type_specific_stats):
        return column_stats
    elif int_stats := std_get_if[integer_statistics](&type_specific_stats):
        if int_stats.minimum.has_value():
            column_stats["minimum"] = int_stats.minimum.value()
        else:
            column_stats["minimum"] = None
        if int_stats.maximum.has_value():
            column_stats["maximum"] = int_stats.maximum.value()
        else:
            column_stats["maximum"] = None
        if int_stats.sum.has_value():
            column_stats["sum"] = int_stats.sum.value()
        else:
            column_stats["sum"] = None
    elif dbl_stats := std_get_if[double_statistics](&type_specific_stats):
        if dbl_stats.minimum.has_value():
            column_stats["minimum"] = dbl_stats.minimum.value()
        else:
            column_stats["minimum"] = None
        if dbl_stats.maximum.has_value():
            column_stats["maximum"] = dbl_stats.maximum.value()
        else:
            column_stats["maximum"] = None
        if dbl_stats.sum.has_value():
            column_stats["sum"] = dbl_stats.sum.value()
        else:
            column_stats["sum"] = None
    elif str_stats := std_get_if[string_statistics](&type_specific_stats):
        if str_stats.minimum.has_value():
            column_stats["minimum"] = str_stats.minimum.value().decode("utf-8")
        else:
            column_stats["minimum"] = None
        if str_stats.maximum.has_value():
            column_stats["maximum"] = str_stats.maximum.value().decode("utf-8")
        else:
            column_stats["maximum"] = None
        if str_stats.sum.has_value():
            column_stats["sum"] = str_stats.sum.value()
        else:
            column_stats["sum"] = None
    elif bucket_stats := std_get_if[bucket_statistics](&type_specific_stats):
        column_stats["true_count"] = bucket_stats.count[0]
        column_stats["false_count"] = (
            column_stats["number_of_values"]
            - column_stats["true_count"]
        )
    elif dec_stats := std_get_if[decimal_statistics](&type_specific_stats):
        if dec_stats.minimum.has_value():
            column_stats["minimum"] = dec_stats.minimum.value().decode("utf-8")
        else:
            column_stats["minimum"] = None
        if dec_stats.maximum.has_value():
            column_stats["maximum"] = dec_stats.maximum.value().decode("utf-8")
        else:
            column_stats["maximum"] = None
        if dec_stats.sum.has_value():
            column_stats["sum"] = dec_stats.sum.value().decode("utf-8")
        else:
            column_stats["sum"] = None
    elif date_stats := std_get_if[date_statistics](&type_specific_stats):
        if date_stats.minimum.has_value():
            column_stats["minimum"] = datetime.datetime.fromtimestamp(
                datetime.timedelta(date_stats.minimum.value()).total_seconds(),
                datetime.timezone.utc,
            )
        else:
            column_stats["minimum"] = None
        if date_stats.maximum.has_value():
            column_stats["maximum"] = datetime.datetime.fromtimestamp(
                datetime.timedelta(date_stats.maximum.value()).total_seconds(),
                datetime.timezone.utc,
            )
        else:
            column_stats["maximum"] = None
    elif bin_stats := std_get_if[binary_statistics](&type_specific_stats):
        if bin_stats.sum.has_value():
            column_stats["sum"] = bin_stats.sum.value()
        else:
            column_stats["sum"] = None
    elif ts_stats := std_get_if[timestamp_statistics](&type_specific_stats):
        # Before ORC-135, the local timezone offset was included and they were
        # stored as minimum and maximum. After ORC-135, the timestamp is
        # adjusted to UTC before being converted to milliseconds and stored
        # in minimumUtc and maximumUtc.
        # TODO: Support minimum and maximum by reading writer's local timezone
        if ts_stats.minimum_utc.has_value() and ts_stats.maximum_utc.has_value():
            column_stats["minimum"] = datetime.datetime.fromtimestamp(
                ts_stats.minimum_utc.value() / 1000, datetime.timezone.utc
            )
            column_stats["maximum"] = datetime.datetime.fromtimestamp(
                ts_stats.maximum_utc.value() / 1000, datetime.timezone.utc
            )
    else:
        raise ValueError("Unsupported statistics type")
    return column_stats


cpdef read_parsed_orc_statistics(filepath_or_buffer):
    """
    Cython function to call into libcudf API, see `read_parsed_orc_statistics`.

    See Also
    --------
    cudf.io.orc.read_orc_statistics
    """

    # Handle NativeFile input
    if isinstance(filepath_or_buffer, NativeFile):
        filepath_or_buffer = NativeFileDatasource(filepath_or_buffer)

    cdef parsed_orc_statistics parsed = (
        libcudf_read_parsed_orc_statistics(make_source_info([filepath_or_buffer]))
    )

    cdef vector[column_statistics] file_stats = parsed.file_stats
    cdef vector[vector[column_statistics]] stripes_stats = parsed.stripes_stats

    parsed_file_stats = [
        _parse_column_type_statistics(file_stats[column_index])
        for column_index in range(file_stats.size())
    ]

    parsed_stripes_stats = [
        [_parse_column_type_statistics(stripes_stats[stripe_index][column_index])
         for column_index in range(stripes_stats[stripe_index].size())]
        for stripe_index in range(stripes_stats.size())
    ]

    return parsed.column_names, parsed_file_stats, parsed_stripes_stats


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
    """
    cdef orc_reader_options c_orc_reader_options = make_orc_reader_options(
        filepaths_or_buffers,
        columns,
        stripes or [],
        get_skiprows_arg(skip_rows),
        get_num_rows_arg(num_rows),
        (
            type_id.EMPTY
            if timestamp_type is None else
            <type_id>(
                <underlying_type_t_type_id> (
                    SUPPORTED_NUMPY_TO_LIBCUDF_TYPES[
                        cudf.dtype(timestamp_type)
                    ]
                )
            )
        ),
        use_index,
    )

    cdef table_with_metadata c_result
    cdef size_type nrows

    with nogil:
        c_result = move(libcudf_read_orc(c_orc_reader_options))

    names = [info.name.decode() for info in c_result.metadata.schema_info]
    actual_index_names, col_names, is_range_index, reset_index_name, \
        range_idx = _get_index_from_metadata(c_result.metadata.user_data,
                                             names,
                                             skip_rows,
                                             num_rows)

    if columns is not None and (isinstance(columns, list) and len(columns) == 0):
        # When `columns=[]`, index needs to be
        # established, but not the columns.
        nrows = c_result.tbl.get()[0].view().num_rows()
        return {}, cudf.RangeIndex(nrows)

    data, index = data_from_unique_ptr(
        move(c_result.tbl),
        col_names if columns is None else names,
        actual_index_names
    )

    if is_range_index:
        index = range_idx
    elif reset_index_name:
        index.names = [None] * len(index.names)

    data = {
        name: update_column_struct_field_names(
            col, c_result.metadata.schema_info[i]
        )
        for i, (name, col) in enumerate(data.items())
    }

    return data, index


cdef compression_type _get_comp_type(object compression):
    if compression is None or compression is False:
        return compression_type.NONE

    compression = str(compression).upper()
    if compression == "SNAPPY":
        return compression_type.SNAPPY
    elif compression == "ZLIB":
        return compression_type.ZLIB
    elif compression == "ZSTD":
        return compression_type.ZSTD
    elif compression == "LZ4":
        return compression_type.LZ4
    else:
        raise ValueError(f"Unsupported `compression` type {compression}")

cdef tuple _get_index_from_metadata(
        map[string, string] user_data,
        object names,
        object skip_rows,
        object num_rows):
    json_str = user_data[b'pandas'].decode('utf-8')
    meta = None
    index_col = None
    is_range_index = False
    reset_index_name = False
    range_idx = None
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

cdef cudf_io_types.statistics_freq _get_orc_stat_freq(object statistics):
    """
    Convert ORC statistics terms to CUDF convention:
      - ORC "STRIPE"   == CUDF "ROWGROUP"
      - ORC "ROWGROUP" == CUDF "PAGE"
    """
    statistics = str(statistics).upper()
    if statistics == "NONE":
        return cudf_io_types.statistics_freq.STATISTICS_NONE
    elif statistics == "STRIPE":
        return cudf_io_types.statistics_freq.STATISTICS_ROWGROUP
    elif statistics == "ROWGROUP":
        return cudf_io_types.statistics_freq.STATISTICS_PAGE
    else:
        raise ValueError(f"Unsupported `statistics_freq` type {statistics}")


@acquire_spill_lock()
def write_orc(
    table,
    object path_or_buf,
    object compression="snappy",
    object statistics="ROWGROUP",
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
    cdef compression_type compression_ = _get_comp_type(compression)
    cdef unique_ptr[data_sink] data_sink_c
    cdef sink_info sink_info_c = make_sink_info(path_or_buf, data_sink_c)
    cdef table_input_metadata tbl_meta
    cdef map[string, string] user_data
    user_data[str.encode("pandas")] = str.encode(generate_pandas_metadata(
        table, index)
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

    if cols_as_map_type is not None:
        cols_as_map_type = set(cols_as_map_type)

    for i, name in enumerate(table._column_names, num_index_cols_meta):
        tbl_meta.column_metadata[i].set_name(name.encode())
        _set_col_children_metadata(
            table[name]._column,
            tbl_meta.column_metadata[i],
            (cols_as_map_type is not None)
            and (name in cols_as_map_type),
        )

    cdef orc_writer_options c_orc_writer_options = move(
        orc_writer_options.builder(
            sink_info_c, tv
        ).metadata(tbl_meta)
        .key_value_metadata(move(user_data))
        .compression(compression_)
        .enable_statistics(_get_orc_stat_freq(statistics))
        .build()
    )
    if stripe_size_bytes is not None:
        c_orc_writer_options.set_stripe_size_bytes(stripe_size_bytes)
    if stripe_size_rows is not None:
        c_orc_writer_options.set_stripe_size_rows(stripe_size_rows)
    if row_index_stride is not None:
        c_orc_writer_options.set_row_index_stride(row_index_stride)

    with nogil:
        libcudf_write_orc(c_orc_writer_options)


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


cdef orc_reader_options make_orc_reader_options(
    object filepaths_or_buffers,
    object column_names,
    object stripes,
    int64_t skip_rows,
    int64_t num_rows,
    type_id timestamp_type,
    bool use_index
) except*:

    for i, datasource in enumerate(filepaths_or_buffers):
        if isinstance(datasource, NativeFile):
            filepaths_or_buffers[i] = NativeFileDatasource(datasource)
    cdef vector[vector[size_type]] strps = stripes
    cdef orc_reader_options opts
    cdef source_info src = make_source_info(filepaths_or_buffers)
    opts = move(
        orc_reader_options.builder(src)
        .stripes(strps)
        .skip_rows(skip_rows)
        .timestamp_type(data_type(timestamp_type))
        .use_index(use_index)
        .build()
    )
    if num_rows >= 0:
        opts.set_num_rows(num_rows)

    cdef vector[string] c_column_names
    if column_names is not None:
        c_column_names.reserve(len(column_names))
        for col in column_names:
            c_column_names.push_back(str(col).encode())
        if len(column_names) > 0:
            opts.set_columns(c_column_names)

    return opts


cdef class ORCWriter:
    """
    ORCWriter lets you you incrementally write out a ORC file from a series
    of cudf tables

    See Also
    --------
    cudf.io.orc.to_orc
    """
    cdef bool initialized
    cdef unique_ptr[orc_chunked_writer] writer
    cdef sink_info sink
    cdef unique_ptr[data_sink] _data_sink
    cdef cudf_io_types.statistics_freq stat_freq
    cdef compression_type comp_type
    cdef object index
    cdef table_input_metadata tbl_meta
    cdef object cols_as_map_type
    cdef object stripe_size_bytes
    cdef object stripe_size_rows
    cdef object row_index_stride

    def __cinit__(self,
                  object path,
                  object index=None,
                  object compression="snappy",
                  object statistics="ROWGROUP",
                  object cols_as_map_type=None,
                  object stripe_size_bytes=None,
                  object stripe_size_rows=None,
                  object row_index_stride=None):

        self.sink = make_sink_info(path, self._data_sink)
        self.stat_freq = _get_orc_stat_freq(statistics)
        self.comp_type = _get_comp_type(compression)
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
        tv = table_view_from_table(table, not keep_index)

        with nogil:
            self.writer.get()[0].write(tv)

    def close(self):
        if not self.initialized:
            return

        with nogil:
            self.writer.get()[0].close()

    def __dealloc__(self):
        self.close()

    def _initialize_chunked_state(self, table):
        """
        Prepare all the values required to build the
        chunked_orc_writer_options anb creates a writer"""
        cdef table_view tv

        num_index_cols_meta = 0
        self.tbl_meta = table_input_metadata(
            table_view_from_table(table, ignore_index=True),
        )
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
            _set_col_children_metadata(
                table[name]._column,
                self.tbl_meta.column_metadata[i],
                (self.cols_as_map_type is not None)
                and (name in self.cols_as_map_type),
            )

        cdef map[string, string] user_data
        pandas_metadata = generate_pandas_metadata(table, self.index)
        user_data[str.encode("pandas")] = str.encode(pandas_metadata)

        cdef chunked_orc_writer_options c_opts = move(
                chunked_orc_writer_options.builder(self.sink)
                .metadata(self.tbl_meta)
                .key_value_metadata(move(user_data))
                .compression(self.comp_type)
                .enable_statistics(self.stat_freq)
                .build()
            )
        if self.stripe_size_bytes is not None:
            c_opts.set_stripe_size_bytes(self.stripe_size_bytes)
        if self.stripe_size_rows is not None:
            c_opts.set_stripe_size_rows(self.stripe_size_rows)
        if self.row_index_stride is not None:
            c_opts.set_row_index_stride(self.row_index_stride)

        with nogil:
            self.writer.reset(new orc_chunked_writer(c_opts))

        self.initialized = True

cdef _set_col_children_metadata(Column col,
                                column_in_metadata& col_meta,
                                list_column_as_map=False):
    if isinstance(col.dtype, cudf.StructDtype):
        for i, (child_col, name) in enumerate(
            zip(col.children, list(col.dtype.fields))
        ):
            col_meta.child(i).set_name(name.encode())
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
