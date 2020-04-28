# Copyright (c) 2020, NVIDIA CORPORATION.

from enum import IntEnum

import cudf
from io import BytesIO, IOBase, StringIO
import os

from libcpp cimport bool

from cudf._lib.cpp.io.functions cimport (
    read_csv as cpp_read_csv,
    read_csv_args
)
from cudf._lib.io.utils cimport make_source_info
from cudf._lib.move cimport move
from cudf._lib.nvtx import annotate
from cudf._lib.table cimport Table
from cudf._lib.cpp.io.types cimport (
    compression_type, 
    quote_style, 
    source_info,
    table_with_metadata
)

from cudf.utils import ioutils

from libc.stdint cimport int32_t


ctypedef int32_t underlying_type_t_compression

class Compression(IntEnum):
    NONE = ( 
        <underlying_type_t_compression> compression_type.NONE
    )
    INFER = (
        <underlying_type_t_compression> compression_type.AUTO
    )
    SNAPPY = (
        <underlying_type_t_compression> compression_type.SNAPPY
    )
    GZIP = (
        <underlying_type_t_compression> compression_type.GZIP
    )
    BZIP2 = (
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
    filepath_or_buffer,
    object lineterminator,
    object quotechar,
    int quoting,
    bool doublequote,
    header,
    bool mangle_dupe_cols,
    usecols,
    object sep,
    object delimiter,
    bool delim_whitespace,
    bool skipinitialspace,
    names,
    dtype,
    int skipfooter,
    int skiprows,
    bool dayfirst,
    compression,
    thousands,
    decimal,
    true_values,
    false_values,
    int nrows,
    bool skip_blank_lines,
    parse_dates,
    comment,
    na_values,
    bool keep_default_na,
    bool na_filter,
    object prefix,
    index_col,
    ) except +:

    cdef source_info c_source_info = make_source_info(filepath_or_buffer)
    cdef read_csv_args read_csv_args_c = read_csv_args(c_source_info)
    read_csv_args_c.lineterminator = ord(lineterminator)
    read_csv_args_c.quotechar = ord(quotechar)
    if quoting == 1:
        read_csv_args_c.quoting = quote_style.QUOTE_ALL
    elif quoting == 2:
        read_csv_args_c.quoting = quote_style.QUOTE_NONNUMERIC
    elif quoting == 3:
        read_csv_args_c.quoting = quote_style.QUOTE_NONE
    else:
        # Default value
        read_csv_args_c.quoting = quote_style.QUOTE_MINIMAL
    read_csv_args_c.doublequote = doublequote
    if header == 'infer':
        read_csv_args_c.header = -1
    else:
        read_csv_args_c.header = 1-int(header)
    read_csv_args_c.mangle_dupe_cols = mangle_dupe_cols
    read_csv_args_c.delim_whitespace = delim_whitespace
    read_csv_args_c.skipinitialspace = skipinitialspace
    read_csv_args_c.skip_blank_lines = skip_blank_lines
    read_csv_args_c.nrows = nrows
    read_csv_args_c.skiprows = skiprows
    read_csv_args_c.skipfooter = skipfooter
    if comment is not None:
        read_csv_args_c.comment = ord(comment)
    read_csv_args_c.decimal = ord(decimal)
    if thousands is not None:
        read_csv_args_c.thousands = ord(thousands)
    if prefix is not None:
        read_csv_args_c.prefix = prefix
    compression = 'none' if compression is None else compression
    compression = Compression[compression.upper()]
    read_csv_args_c.compression = <compression_type> (
        <underlying_type_t_compression> compression
    )
    if usecols is not None and len(usecols) > 0:
        if isinstance(usecols[0], str):
            read_csv_args_c.use_cols_names.reserve(len(usecols))
            for name in usecols:
                read_csv_args_c.use_cols_names.push_back(name.encode())
        else:
            read_csv_args_c.use_cols_indexes.reserve(len(usecols))
            for index in usecols:
                read_csv_args_c.use_cols_indexes.push_back(index)

    if names is not None and len(names) > 0:
        read_csv_args_c.names.reserve(len(names))
        for name in names:
            read_csv_args_c.names.push_back(name.encode())

    if dtype is not None and len(dtype) > 0:
        read_csv_args_c.dtype.reserve(len(dtype))
        for dt in dtype:
            read_csv_args_c.dtype.push_back(str(dt).encode())

    if true_values is not None and len(true_values) > 0:
        read_csv_args_c.true_values.reserve(len(true_values))
        for tv in true_values:
            read_csv_args_c.true_values.push_back(tv.encode())

    if false_values is not None and len(false_values) > 0:
        read_csv_args_c.false_values.reserve(len(false_values))
        for fv in false_values:
            read_csv_args_c.false_values.push_back(fv.encode())
    
    if na_values is not None and len(na_values) > 0:
        read_csv_args_c.na_values.reserve(len(na_values))
        for nv in na_values:
            read_csv_args_c.na_values.push_back(nv.encode())
    
    if sep is not None:
        read_csv_args_c.delimiter = ord(sep)
    elif delimiter is not None:
        read_csv_args_c.delimiter = ord(delimiter)

    if parse_dates is not None and len(parse_dates) > 0:
        if isinstance(parse_dates[0], str):
            for name in parse_dates:
                print(" RGSL --------------- name is ", parse_dates[0])
                read_csv_args_c.infer_date_names.push_back(name.encode())
        else:
            for index in parse_dates:
                read_csv_args_c.infer_date_indexes.push_back(index)

    read_csv_args_c.dayfirst=dayfirst

    return read_csv_args_c

def read_csv(
    filepath_or_buffer,
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
    nrows=-1,
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
    """{docstring}"""

    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        filepath_or_buffer, compression, (BytesIO, StringIO), **kwargs
    )

    if isinstance(filepath_or_buffer, StringIO):
        filepath_or_buffer = filepath_or_buffer.read().encode()
    elif isinstance(filepath_or_buffer, str) and not os.path.isfile(filepath_or_buffer):
        filepath_or_buffer = filepath_or_buffer.encode()

    cdef read_csv_args read_csv_arg_c = make_read_csv_args(filepath_or_buffer,
        lineterminator, quotechar, quoting, doublequote, header,
        mangle_dupe_cols, usecols, sep, delimiter, delim_whitespace,
        skipinitialspace,  names, dtype, skipfooter, skiprows, dayfirst,
        compression, thousands, decimal, true_values, false_values, nrows,
        skip_blank_lines, parse_dates, comment, na_values, keep_default_na,
        na_filter, prefix, index_col)


    cdef table_with_metadata c_result
    with nogil:
        c_result = move(cpp_read_csv(read_csv_arg_c))

    names = [name.decode() for name in c_result.metadata.column_names]
    return cudf.DataFrame._from_table(Table.from_unique_ptr(
        move(c_result.tbl),
        column_names=names
    ))




