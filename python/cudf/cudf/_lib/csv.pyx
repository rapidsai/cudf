# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string

import cudf

import collections.abc as abc
import errno
from io import BytesIO, StringIO
import os

from enum import IntEnum

from libcpp cimport bool

from libc.stdint cimport int32_t

from cudf._lib.cpp.io.functions cimport (
    read_csv as cpp_read_csv,
    read_csv_args,
    write_csv as cpp_write_csv,
    write_csv_args
)
from cudf._lib.cpp.io.types cimport (
    compression_type,
    data_sink,
    quote_style,
    sink_info,
    source_info,
    table_metadata,
    table_with_metadata
)
from cudf._lib.io.utils cimport make_source_info, make_sink_info
from cudf._lib.move cimport move
from cudf._lib.table cimport Table
from cudf._lib.cpp.table.table_view cimport table_view

ctypedef int32_t underlying_type_t_compression


class Compression(IntEnum):
    INFER = (
        <underlying_type_t_compression> compression_type.AUTO
    )
    SNAPPY = (
        <underlying_type_t_compression> compression_type.SNAPPY
    )
    GZIP = (
        <underlying_type_t_compression> compression_type.GZIP
    )
    BZ2 = (
        <underlying_type_t_compression> compression_type.BZIP2
    )
    BROTLI = (
        <underlying_type_t_compression> compression_type.BROTLI
    )
    ZIP = (
        <underlying_type_t_compression> compression_type.ZIP
    )
    XZ = (
        <underlying_type_t_compression> compression_type.XZ
    )


cdef read_csv_args make_read_csv_args(
    object datasource,
    object lineterminator,
    object quotechar,
    int quoting,
    bool doublequote,
    object header,
    bool mangle_dupe_cols,
    object usecols,
    object delimiter,
    bool delim_whitespace,
    bool skipinitialspace,
    object names,
    object dtype,
    int skipfooter,
    int skiprows,
    bool dayfirst,
    object compression,
    object thousands,
    object decimal,
    object true_values,
    object false_values,
    object nrows,
    object byte_range,
    bool skip_blank_lines,
    object parse_dates,
    object comment,
    object na_values,
    bool keep_default_na,
    bool na_filter,
    object prefix,
    object index_col,
) except +:
    cdef source_info c_source_info = make_source_info([datasource])
    cdef read_csv_args read_csv_args_c = read_csv_args(c_source_info)

    # Reader settings
    if compression is None:
        read_csv_args_c.compression = compression_type.NONE
    else:
        compression = Compression[compression.upper()]
        read_csv_args_c.compression = <compression_type> (
            <underlying_type_t_compression> compression
        )

    if names is not None:
        # explicitly mentioned name, so don't check header
        if header is None or header == 'infer':
            read_csv_args_c.header = -1
        else:
            read_csv_args_c.header = header

        read_csv_args_c.names.reserve(len(names))
        for name in names:
            read_csv_args_c.names.push_back(str(name).encode())
    else:
        if header is -1:
            header_infer = 0
        if header is None:
            header_infer = -1

    if prefix is not None:
        read_csv_args_c.prefix = prefix.encode()

    read_csv_args_c.mangle_dupe_cols = mangle_dupe_cols
    read_csv_args_c.byte_range_offset = (
        byte_range[0] if byte_range is not None else 0
    )
    read_csv_args_c.byte_range_size = (
        byte_range[1] if byte_range is not None else 0
    )

    # Filter settings
    if usecols is not None:
        all_int = all(isinstance(col, int) for col in usecols)
        if all_int:
            read_csv_args_c.use_cols_indexes.reserve(len(usecols))
            read_csv_args_c.use_cols_indexes = usecols
        else:
            read_csv_args_c.use_cols_names.reserve(len(usecols))
            for col_name in usecols:
                read_csv_args_c.use_cols_names.push_back(
                    str(col_name).encode()
                )

    if names is None:
        if header is None:
            read_csv_args_c.header = -1
        elif header == 'infer':
            read_csv_args_c.header = 0
        else:
            read_csv_args_c.header = header

    read_csv_args_c.nrows = nrows if nrows is not None else -1
    read_csv_args_c.skiprows = skiprows
    read_csv_args_c.skipfooter = skipfooter

    # Parsing settings
    if delimiter is not None:
        read_csv_args_c.delimiter = ord(delimiter)

    if thousands is not None:
        read_csv_args_c.thousands = ord(thousands)

    if comment is not None:
        read_csv_args_c.comment = ord(comment)

    if quoting == 1:
        read_csv_args_c.quoting = quote_style.QUOTE_ALL
    elif quoting == 2:
        read_csv_args_c.quoting = quote_style.QUOTE_NONNUMERIC
    elif quoting == 3:
        read_csv_args_c.quoting = quote_style.QUOTE_NONE
    else:
        # Default value
        read_csv_args_c.quoting = quote_style.QUOTE_MINIMAL

    if parse_dates is not None:
        if isinstance(parse_dates, abc.Mapping):
            raise NotImplementedError(
                "`parse_dates`: dictionaries are unsupported")
        if not isinstance(parse_dates, abc.Iterable):
            raise NotImplementedError(
                "`parse_dates`: non-lists are unsupported")
        for col in parse_dates:
            if isinstance(col, str):
                read_csv_args_c.infer_date_names.push_back(str(col).encode())
            elif isinstance(col, int):
                read_csv_args_c.infer_date_indexes.push_back(col)
            else:
                raise NotImplementedError(
                    "`parse_dates`: Nesting is unsupported")

    read_csv_args_c.lineterminator = ord(lineterminator)
    read_csv_args_c.quotechar = ord(quotechar)
    read_csv_args_c.decimal = ord(decimal)
    read_csv_args_c.delim_whitespace = delim_whitespace
    read_csv_args_c.skipinitialspace = skipinitialspace
    read_csv_args_c.skip_blank_lines = skip_blank_lines
    read_csv_args_c.doublequote = doublequote

    # Conversion settings
    if dtype is not None:
        if isinstance(dtype, abc.Mapping):
            read_csv_args_c.dtype.reserve(len(dtype))
            for k, v in dtype.items():
                read_csv_args_c.dtype.push_back(
                    str(str(k)+":"+str(v)).encode()
                )
        elif isinstance(dtype, abc.Iterable):
            read_csv_args_c.dtype.reserve(len(dtype))
            for col_dtype in dtype:
                read_csv_args_c.dtype.push_back(str(col_dtype).encode())
        else:
            read_csv_args_c.dtype.push_back(str(dtype).encode())

    if true_values is not None:
        read_csv_args_c.true_values.reserve(len(true_values))
        for tv in true_values:
            read_csv_args_c.true_values.push_back(tv.encode())

    if false_values is not None:
        read_csv_args_c.false_values.reserve(len(false_values))
        for fv in false_values:
            read_csv_args_c.false_values.push_back(fv.encode())

    if na_values is not None:
        read_csv_args_c.na_values.reserve(len(na_values))
        for nv in na_values:
            read_csv_args_c.na_values.push_back(nv.encode())

    read_csv_args_c.keep_default_na = keep_default_na
    read_csv_args_c.na_filter = na_filter
    read_csv_args_c.dayfirst=dayfirst

    return read_csv_args_c


def validate_args(
    delimiter,
    sep,
    delim_whitespace,
    decimal,
    thousands,
    nrows,
    skipfooter,
    byte_range,
    skiprows
):
    if delim_whitespace:
        if delimiter is not None:
            raise ValueError("cannot set both delimiter and delim_whitespace")
        if sep != ',':
            raise ValueError("cannot set both sep and delim_whitespace")

    # Alias sep -> delimiter.
    actual_delimiter = delimiter if delimiter else sep

    if decimal == actual_delimiter:
        raise ValueError("decimal cannot be the same as delimiter")

    if thousands == actual_delimiter:
        raise ValueError("thousands cannot be the same as delimiter")

    if nrows is not None and skipfooter != 0:
        raise ValueError("cannot use both nrows and skipfooter parameters")

    if byte_range is not None:
        if skipfooter != 0 or skiprows != 0 or nrows is not None:
            raise ValueError("""cannot manually limit rows to be read when
                                using the byte range parameter""")


def read_csv(
    datasource,
    lineterminator="\n",
    quotechar='"',
    quoting=0,
    doublequote=True,
    header="infer",
    mangle_dupe_cols=True,
    usecols=None,
    sep=",",
    delimiter=None,
    delim_whitespace=False,
    skipinitialspace=False,
    names=None,
    dtype=None,
    skipfooter=0,
    skiprows=0,
    dayfirst=False,
    compression="infer",
    thousands=None,
    decimal=".",
    true_values=None,
    false_values=None,
    nrows=None,
    byte_range=None,
    skip_blank_lines=True,
    parse_dates=None,
    comment=None,
    na_values=None,
    keep_default_na=True,
    na_filter=True,
    prefix=None,
    index_col=None,
    **kwargs,
):
    """
    Cython function to call into libcudf API, see `read_csv`.

    See Also
    --------
    cudf.io.csv.read_csv
    """

    if not isinstance(datasource, (BytesIO, StringIO, bytes,
                                   cudf._lib.io.datasource.Datasource)):
        if not os.path.isfile(datasource):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), datasource
            )

    if isinstance(datasource, StringIO):
        datasource = datasource.read().encode()
    elif isinstance(datasource, str) and not os.path.isfile(datasource):
        datasource = datasource.encode()

    validate_args(delimiter, sep, delim_whitespace, decimal, thousands,
                  nrows, skipfooter, byte_range, skiprows)

    # Alias sep -> delimiter.
    if delimiter is None:
        delimiter = sep

    cdef read_csv_args read_csv_arg_c = make_read_csv_args(
        datasource, lineterminator, quotechar, quoting, doublequote,
        header, mangle_dupe_cols, usecols, delimiter, delim_whitespace,
        skipinitialspace, names, dtype, skipfooter, skiprows, dayfirst,
        compression, thousands, decimal, true_values, false_values, nrows,
        byte_range, skip_blank_lines, parse_dates, comment, na_values,
        keep_default_na, na_filter, prefix, index_col)

    cdef table_with_metadata c_result
    with nogil:
        c_result = move(cpp_read_csv(read_csv_arg_c))

    meta_names = [name.decode() for name in c_result.metadata.column_names]
    df = cudf.DataFrame._from_table(Table.from_unique_ptr(
        move(c_result.tbl),
        column_names=meta_names
    ))

    if names is not None and isinstance(names[0], (int)):
        df.columns = [int(x) for x in df._data]

    # Set index if the index_col parameter is passed
    if index_col is not None and index_col is not False:
        if isinstance(index_col, int):
            df = df.set_index(df._data.select_by_index(index_col).names[0])
            if names is None:
                df._index.name = index_col
        else:
            df = df.set_index(index_col)

    return df


cpdef write_csv(
    Table table,
    path_or_buf=None,
    sep=",",
    na_rep="",
    header=True,
    line_terminator="\n",
    rows_per_chunk=8,

):
    """
    Cython function to call into libcudf API, see `write_csv`.

    See Also
    --------
    cudf.io.csv.write_csv
    """

    # Index already been reset and added as main column, so just data_view
    cdef table_view input_table_view = table.data_view()
    cdef bool include_header_c = header
    cdef char delim_c = ord(sep)
    cdef string line_term_c = line_terminator.encode()
    cdef string na_c = na_rep.encode()
    cdef int rows_per_chunk_c = rows_per_chunk
    cdef table_metadata metadata_ = table_metadata()
    cdef string true_value_c = 'True'.encode()
    cdef string false_value_c = 'False'.encode()
    cdef unique_ptr[data_sink] data_sink_c
    cdef sink_info sink_info_c = make_sink_info(path_or_buf, &data_sink_c)

    if header is True and table._column_names is not None:
        metadata_.column_names.reserve(len(table._column_names))
        for col_name in table._column_names:
            metadata_.column_names.push_back(str(col_name).encode())

    cdef unique_ptr[write_csv_args] write_csv_args_c = (
        make_unique[write_csv_args](
            sink_info_c, input_table_view, na_c, include_header_c,
            rows_per_chunk_c, line_term_c, delim_c, true_value_c,
            false_value_c, &metadata_
        )
    )

    with nogil:
        cpp_write_csv(write_csv_args_c.get()[0])
