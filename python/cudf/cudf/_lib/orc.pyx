# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import cudf

from libcpp cimport bool, int
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.io.orc cimport (
    chunked_orc_writer_options,
    orc_chunked_writer,
    orc_reader_options,
    orc_writer_options,
    read_orc as libcudf_read_orc,
    write_orc as libcudf_write_orc,
)
from cudf._lib.cpp.io.orc_metadata cimport (
    raw_orc_statistics,
    read_raw_orc_statistics as libcudf_read_raw_orc_statistics,
)
from cudf._lib.cpp.io.types cimport (
    column_in_metadata,
    column_name_info,
    compression_type,
    data_sink,
    sink_info,
    source_info,
    table_input_metadata,
    table_with_metadata,
)
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport data_type, size_type, type_id
from cudf._lib.io.datasource cimport NativeFileDatasource
from cudf._lib.io.utils cimport (
    make_sink_info,
    make_source_info,
    update_column_struct_field_names,
    update_struct_field_names,
)

from cudf._lib.types import SUPPORTED_NUMPY_TO_LIBCUDF_TYPES

from cudf._lib.types cimport underlying_type_t_type_id

import numpy as np

from cudf._lib.utils cimport (
    data_from_unique_ptr,
    get_column_names,
    table_view_from_table,
)

from pyarrow.lib import NativeFile

from cudf._lib.utils import _index_level_name, generate_pandas_metadata
from cudf.api.types import is_list_dtype, is_struct_dtype


cpdef read_raw_orc_statistics(filepath_or_buffer):
    """
    Cython function to call into libcudf API, see `read_raw_orc_statistics`.

    See Also
    --------
    cudf.io.orc.read_orc_statistics
    """

    # Handle NativeFile input
    if isinstance(filepath_or_buffer, NativeFile):
        filepath_or_buffer = NativeFileDatasource(filepath_or_buffer)

    cdef raw_orc_statistics raw = (
        libcudf_read_raw_orc_statistics(make_source_info([filepath_or_buffer]))
    )
    return (raw.column_names, raw.file_stats, raw.stripes_stats)


cpdef read_orc(object filepaths_or_buffers,
               object columns=None,
               object stripes=None,
               object skip_rows=None,
               object num_rows=None,
               bool use_index=True,
               object decimal_cols_as_float=None,
               object timestamp_type=None):
    """
    Cython function to call into libcudf API, see `read_orc`.

    See Also
    --------
    cudf.read_orc
    """
    cdef orc_reader_options c_orc_reader_options = make_orc_reader_options(
        filepaths_or_buffers,
        columns or [],
        stripes or [],
        get_size_t_arg(skip_rows, "skip_rows"),
        get_size_t_arg(num_rows, "num_rows"),
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
        decimal_cols_as_float or [],
    )

    cdef table_with_metadata c_result

    with nogil:
        c_result = move(libcudf_read_orc(c_orc_reader_options))

    names = [name.decode() for name in c_result.metadata.column_names]

    data, index = data_from_unique_ptr(move(c_result.tbl), names)

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
    elif compression == "snappy":
        return compression_type.SNAPPY
    else:
        raise ValueError(f"Unsupported `compression` type {compression}")


cpdef write_orc(table,
                object path_or_buf,
                object compression=None,
                bool enable_statistics=True,
                object stripe_size_bytes=None,
                object stripe_size_rows=None,
                object row_index_stride=None):
    """
    Cython function to call into libcudf API, see `write_orc`.

    See Also
    --------
    cudf.read_orc
    """
    cdef compression_type compression_ = _get_comp_type(compression)
    cdef unique_ptr[data_sink] data_sink_c
    cdef sink_info sink_info_c = make_sink_info(path_or_buf, data_sink_c)
    cdef unique_ptr[table_input_metadata] tbl_meta

    if not isinstance(table._index, cudf.RangeIndex):
        tv = table_view_from_table(table)
        tbl_meta = make_unique[table_input_metadata](tv)
        for level, idx_name in enumerate(table._index.names):
            tbl_meta.get().column_metadata[level].set_name(
                str.encode(
                    _index_level_name(idx_name, level, table._column_names)
                )
            )
        num_index_cols_meta = len(table._index.names)
    else:
        tv = table_view_from_table(table, ignore_index=True)
        tbl_meta = make_unique[table_input_metadata](tv)
        num_index_cols_meta = 0

    for i, name in enumerate(table._column_names, num_index_cols_meta):
        tbl_meta.get().column_metadata[i].set_name(name.encode())
        _set_col_children_names(
            table[name]._column, tbl_meta.get().column_metadata[i]
        )

    cdef orc_writer_options c_orc_writer_options = move(
        orc_writer_options.builder(
            sink_info_c, table_view_from_table(table, ignore_index=True)
        ).metadata(tbl_meta.get())
        .compression(compression_)
        .enable_statistics(<bool> (True if enable_statistics else False))
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


cdef size_type get_size_t_arg(object arg, str name) except*:
    if name == "skip_rows":
        arg = 0 if arg is None else arg
        if not isinstance(arg, int) or arg < 0:
            raise TypeError(f"{name} must be an int >= 0")
    else:
        arg = -1 if arg is None else arg
        if not isinstance(arg, int) or arg < -1:
            raise TypeError(f"{name} must be an int >= -1")
    return <size_type> arg


cdef orc_reader_options make_orc_reader_options(
    object filepaths_or_buffers,
    object column_names,
    object stripes,
    size_type skip_rows,
    size_type num_rows,
    type_id timestamp_type,
    bool use_index,
    object decimal_cols_as_float
) except*:

    for i, datasource in enumerate(filepaths_or_buffers):
        if isinstance(datasource, NativeFile):
            filepaths_or_buffers[i] = NativeFileDatasource(datasource)
    cdef vector[string] c_column_names
    cdef vector[vector[size_type]] strps = stripes
    c_column_names.reserve(len(column_names))
    for col in column_names:
        c_column_names.push_back(str(col).encode())
    cdef orc_reader_options opts
    cdef source_info src = make_source_info(filepaths_or_buffers)
    cdef vector[string] c_decimal_cols_as_float
    c_decimal_cols_as_float.reserve(len(decimal_cols_as_float))
    for decimal_col in decimal_cols_as_float:
        c_decimal_cols_as_float.push_back(str(decimal_col).encode())
    opts = move(
        orc_reader_options.builder(src)
        .columns(c_column_names)
        .stripes(strps)
        .skip_rows(skip_rows)
        .num_rows(num_rows)
        .timestamp_type(data_type(timestamp_type))
        .use_index(use_index)
        .decimal_cols_as_float(c_decimal_cols_as_float)
        .build()
    )

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
    cdef bool enable_stats
    cdef compression_type comp_type
    cdef object index
    cdef unique_ptr[table_input_metadata] tbl_meta

    def __cinit__(self, object path, object index=None,
                  object compression=None, bool enable_statistics=True):
        self.sink = make_sink_info(path, self._data_sink)
        self.enable_stats = enable_statistics
        self.comp_type = _get_comp_type(compression)
        self.index = index
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

        # Set the table_metadata
        num_index_cols_meta = 0
        self.tbl_meta = make_unique[table_input_metadata](
            table_view_from_table(table, ignore_index=True)
        )
        if self.index is not False:
            if isinstance(table._index, cudf.core.multiindex.MultiIndex):
                tv = table_view_from_table(table)
                self.tbl_meta = make_unique[table_input_metadata](tv)
                for level, idx_name in enumerate(table._index.names):
                    self.tbl_meta.get().column_metadata[level].set_name(
                        (str.encode(idx_name))
                    )
                num_index_cols_meta = len(table._index.names)
            else:
                if table._index.name is not None:
                    tv = table_view_from_table(table)
                    self.tbl_meta = make_unique[table_input_metadata](tv)
                    self.tbl_meta.get().column_metadata[0].set_name(
                        str.encode(table._index.name)
                    )
                    num_index_cols_meta = 1

        for i, name in enumerate(table._column_names, num_index_cols_meta):
            self.tbl_meta.get().column_metadata[i].set_name(name.encode())
            _set_col_children_names(
                table[name]._column, self.tbl_meta.get().column_metadata[i]
            )

        pandas_metadata = generate_pandas_metadata(table, self.index)
        self.tbl_meta.get().user_data[str.encode("pandas")] = \
            str.encode(pandas_metadata)

        cdef chunked_orc_writer_options args
        with nogil:
            args = move(
                chunked_orc_writer_options.builder(self.sink)
                .metadata(self.tbl_meta.get())
                .compression(self.comp_type)
                .enable_statistics(self.enable_stats)
                .build()
            )
            self.writer.reset(new orc_chunked_writer(args))

        self.initialized = True

cdef _set_col_children_names(Column col, column_in_metadata& col_meta):
    if is_struct_dtype(col):
        for i, (child_col, name) in enumerate(
            zip(col.children, list(col.dtype.fields))
        ):
            col_meta.child(i).set_name(name.encode())
            _set_col_children_names(child_col, col_meta.child(i))
    elif is_list_dtype(col):
        _set_col_children_names(col.children[1], col_meta.child(1))
    else:
        return
