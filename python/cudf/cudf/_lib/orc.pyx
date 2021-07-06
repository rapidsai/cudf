# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import cudf

from libcpp cimport bool, int
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move
from cudf._lib.cpp.column.column cimport column

from cudf.utils.dtypes import is_struct_dtype

from cudf._lib.column cimport Column

from cudf._lib.cpp.io.orc_metadata cimport (
    raw_orc_statistics,
    read_raw_orc_statistics as libcudf_read_raw_orc_statistics
)
from cudf._lib.cpp.io.orc cimport (
    orc_reader_options,
    read_orc as libcudf_read_orc,
    orc_writer_options,
    write_orc as libcudf_write_orc,
    chunked_orc_writer_options,
    orc_chunked_writer
)
from cudf._lib.cpp.io.types cimport (
    column_name_info,
    compression_type,
    data_sink,
    sink_info,
    source_info,
    table_metadata,
    table_with_metadata,
    table_metadata_with_nullability
)

from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport (
    data_type, type_id, size_type
)

from cudf._lib.io.utils cimport (
    make_source_info,
    make_sink_info,
    update_struct_field_names,
)
from cudf._lib.table cimport Table
from cudf._lib.types import np_to_cudf_types
from cudf._lib.types cimport underlying_type_t_type_id
import numpy as np

from cudf._lib.utils cimport get_column_names

from cudf._lib.utils import (
    _index_level_name,
    generate_pandas_metadata,
)


cpdef read_raw_orc_statistics(filepath_or_buffer):
    """
    Cython function to call into libcudf API, see `read_raw_orc_statistics`.

    See Also
    --------
    cudf.io.orc.read_orc_statistics
    """

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
    cudf.io.orc.read_orc
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
                    np_to_cudf_types[np.dtype(timestamp_type)]
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

    tbl = Table.from_unique_ptr(move(c_result.tbl), names)

    update_struct_field_names(tbl, c_result.metadata.schema_info)

    return tbl


cdef compression_type _get_comp_type(object compression):
    if compression is None or compression is False:
        return compression_type.NONE
    elif compression == "snappy":
        return compression_type.SNAPPY
    else:
        raise ValueError(f"Unsupported `compression` type {compression}")


cpdef write_orc(Table table,
                object path_or_buf,
                object compression=None,
                bool enable_statistics=True):
    """
    Cython function to call into libcudf API, see `write_orc`.

    See Also
    --------
    cudf.io.orc.read_orc
    """
    cdef compression_type compression_ = _get_comp_type(compression)
    cdef table_metadata metadata_ = table_metadata()
    cdef unique_ptr[data_sink] data_sink_c
    cdef sink_info sink_info_c = make_sink_info(path_or_buf, data_sink_c)

    metadata_.column_names.reserve(len(table._column_names))

    for col_name in table._column_names:
        metadata_.column_names.push_back(str.encode(col_name))

    cdef orc_writer_options c_orc_writer_options = move(
        orc_writer_options.builder(sink_info_c, table.data_view())
        .metadata(&metadata_)
        .compression(compression_)
        .enable_statistics(<bool> (True if enable_statistics else False))
        .build()
    )

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

    def __cinit__(self, object path, object index=None,
                  object compression=None, bool enable_statistics=True):
        self.sink = make_sink_info(path, self._data_sink)
        self.enable_stats = enable_statistics
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

    def close(self):
        if not self.initialized:
            return

        with nogil:
            self.writer.get()[0].close()

    def __dealloc__(self):
        self.close()

    def _initialize_chunked_state(self, Table table):
        """
        Prepare all the values required to build the
        chunked_orc_writer_options anb creates a writer"""
        cdef unique_ptr[table_metadata_with_nullability] tbl_meta
        tbl_meta = make_unique[table_metadata_with_nullability]()

        # Set the table_metadata
        tbl_meta.get().column_names = get_column_names(table, self.index)
        pandas_metadata = generate_pandas_metadata(table, self.index)
        tbl_meta.get().user_data[str.encode("pandas")] = \
            str.encode(pandas_metadata)

        cdef chunked_orc_writer_options args
        with nogil:
            args = move(
                chunked_orc_writer_options.builder(self.sink)
                .metadata(tbl_meta.get())
                .compression(self.comp_type)
                .enable_statistics(self.enable_stats)
                .build()
            )
            self.writer.reset(new orc_chunked_writer(args))

        self.initialized = True
