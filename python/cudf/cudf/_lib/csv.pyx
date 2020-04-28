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

import collections.abc as abc
import errno
import os

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
    nrows,
    byte_range,
    bool skip_blank_lines,
    parse_dates,
    comment,
    na_values,
    bool keep_default_na,
    bool na_filter,
    object prefix,
    index_col,
    ) except +:

    if delim_whitespace:
        if delimiter is not None:
            raise ValueError("cannot set both delimiter and delim_whitespace")
        if sep != ',':
            raise ValueError("cannot set both sep and delim_whitespace")

    # Alias sep -> delimiter.
    if delimiter is None:
        delimiter = sep

    if decimal == delimiter:
        raise ValueError("decimal cannot be the same as delimiter")

    if thousands == delimiter:
        raise ValueError("thousands cannot be the same as delimiter")


    if nrows is not None and skipfooter != 0:
        raise ValueError("cannot use both nrows and skipfooter parameters")

    if byte_range is not None:
        if skipfooter != 0 or skiprows != 0 or nrows is not None:
            raise ValueError("""cannot manually limit rows to be read when
                                using the byte range parameter""")

    cdef source_info c_source_info = make_source_info(filepath_or_buffer)
    cdef read_csv_args read_csv_args_c = read_csv_args(c_source_info)
    read_csv_args_c.lineterminator = ord(lineterminator)
    read_csv_args_c.quotechar = ord(quotechar)
    if quoting == 1:
        print("RGSl : Setting quoting to ALL")
        read_csv_args_c.quoting = quote_style.QUOTE_ALL
    elif quoting == 2:
        read_csv_args_c.quoting = quote_style.QUOTE_NONNUMERIC
    elif quoting == 3:
        read_csv_args_c.quoting = quote_style.QUOTE_NONE
    else:
        # Default value
        read_csv_args_c.quoting = quote_style.QUOTE_MINIMAL
    read_csv_args_c.doublequote = doublequote
    if header is None:
        read_csv_args_c.header = -1
    elif header == 'infer':
        read_csv_args_c.header = 0
    else:
        read_csv_args_c.header = header
    read_csv_args_c.mangle_dupe_cols = mangle_dupe_cols
    read_csv_args_c.delim_whitespace = delim_whitespace
    read_csv_args_c.skipinitialspace = skipinitialspace
    read_csv_args_c.skip_blank_lines = skip_blank_lines
    read_csv_args_c.nrows = nrows if nrows is not None else -1
    read_csv_args_c.byte_range_offset = byte_range[0] if byte_range is not None else 0
    read_csv_args_c.byte_range_size = byte_range[1] if byte_range is not None else 0
    read_csv_args_c.skiprows = skiprows
    read_csv_args_c.skipfooter = skipfooter
    read_csv_args_c.keep_default_na = keep_default_na
    read_csv_args_c.na_filter = na_filter
    print ("RGSL : Setting keep_default_na to ", keep_default_na)
    if comment is not None:
        read_csv_args_c.comment = ord(comment)
    read_csv_args_c.decimal = ord(decimal)
    if thousands is not None:
        read_csv_args_c.thousands = ord(thousands)
    if prefix is not None:
        read_csv_args_c.prefix = prefix.encode()
    compression = 'none' if compression is None else compression
    compression = Compression[compression.upper()]
    read_csv_args_c.compression = <compression_type> (
        <underlying_type_t_compression> compression
    )

    if usecols is not None:
        all_int = True
        # TODO Refactor to use `all_of()`
        for col in usecols:
            if not isinstance(col, int):
                all_int = False
                break
        if all_int:
            read_csv_args_c.use_cols_indexes.reserve(len(usecols))
            read_csv_args_c.use_cols_indexes = usecols
        else:
            for col_name in usecols:
                read_csv_args_c.use_cols_names.reserve(len(usecols))
                read_csv_args_c.use_cols_names.push_back(str(col_name).encode())

    """
    if usecols is not None and len(usecols) > 0:
        if isinstance(usecols[0], str):
            read_csv_args_c.use_cols_names.reserve(len(usecols))
            for name in usecols:
                read_csv_args_c.use_cols_names.push_back(name.encode())
        else:
            read_csv_args_c.use_cols_indexes.reserve(len(usecols))
            for index in usecols:
                read_csv_args_c.use_cols_indexes.push_back(index)
    """


    
    names_from_dtype = None
    """
    if dtype is not None:
        if len(dtype) > 0:
            if isinstance(dtype, abc.Mapping):
                names_from_dtype = [None]*len(dtype.keys())
                dtypes = [None]*len(dtype.values())
                for idx, item in enumerate(dtype.items()):
                    names_from_dtype[idx] = item[0]
                    dtypes[idx] = item[1]
            else:
                dtypes = dtype
            read_csv_args_c.dtype.reserve(len(dtypes))
            for dt in dtype:
                read_csv_args_c.dtype.push_back(str(dt).encode())
        else:
            read_csv_args_c.dtype.reserve(1)
            read_csv_args_c.dtype.push_back(str(dtype).encode())
    """
    if dtype is not None:
        if isinstance(dtype, abc.Mapping):
            for k, v in dtype.items():
                read_csv_args_c.dtype.push_back(str(str(k)+":"+str(v)).encode())
        elif isinstance(dtype, abc.Iterable):
            for col_dtype in dtype:
                read_csv_args_c.dtype.push_back(str(col_dtype).encode())
        else:
            read_csv_args_c.dtype.push_back(str(dtype).encode())
    
    if names is not None:
        # explicitly mentioned name, so don't check header
        read_csv_args_c.header = -1
        read_csv_args_c.names.reserve(len(names))
        for name in names:
            read_csv_args_c.names.push_back(str(name).encode())
    elif names_from_dtype is not None:
        for name in names_from_dtype:
            read_csv_args_c.names.push_back(name.encode())

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
    
    if delimiter is not None:
        read_csv_args_c.delimiter = ord(delimiter)

    if parse_dates is not None and len(parse_dates) > 0:
            for idx in parse_dates:
                if isinstance(idx, str):
                    read_csv_args_c.infer_date_names.push_back(idx.encode())
                else:
                    read_csv_args_c.infer_date_indexes.push_back(idx)

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
    """{docstring}"""


    
    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        filepath_or_buffer, compression, (BytesIO, StringIO), **kwargs
    )
    
    if not isinstance(filepath_or_buffer, (BytesIO, StringIO, bytes)):
        if not os.path.isfile(filepath_or_buffer):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filepath_or_buffer
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
        byte_range, skip_blank_lines, parse_dates, comment, na_values, keep_default_na,
        na_filter, prefix, index_col)


    cdef table_with_metadata c_result
    with nogil:
        c_result = move(cpp_read_csv(read_csv_arg_c))

    meta_names = [name.decode() for name in c_result.metadata.column_names]
    df = cudf.DataFrame._from_table(Table.from_unique_ptr(
        move(c_result.tbl),
        column_names=meta_names
    ))


    if names is not None and isinstance(names[0], (int)):
        df.columns = [int(x) for x in df.columns]

    # Set index if the index_col parameter is passed
    if index_col is not None and index_col is not False:
        if isinstance(index_col, int):
            df = df.set_index(df.columns[index_col])
        else:
            df = df.set_index(index_col)

    return df






