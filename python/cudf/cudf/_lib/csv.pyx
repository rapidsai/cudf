# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._lib.utils cimport *
from cudf._lib.utils import *
from cudf._lib.nvtx import nvtx_range_push, nvtx_range_pop
from cudf._lib.includes.csv cimport (
    reader as csv_reader,
    reader_options as csv_reader_options
)
from libc.stdlib cimport free
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._lib.includes.io cimport *

import numpy as np
import collections.abc as abc

import errno
import os

cimport cudf._lib.includes.csv as cpp_csv


_quoting_enum = {
    0: cpp_csv.QUOTE_MINIMAL,
    1: cpp_csv.QUOTE_ALL,
    2: cpp_csv.QUOTE_NONNUMERIC,
    3: cpp_csv.QUOTE_NONE,
}


cpdef read_csv(
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
    comment=None,
    parse_dates=None,
    na_values=None,
    keep_default_na=True,
    na_filter=True,
    prefix=None,
    index_col=None,
):
    """
    Cython function to call into libcudf API, see `read_csv`.

    See Also
    --------
    cudf.io.csv.read_csv
    """

    if delim_whitespace:
        if delimiter is not None:
            raise ValueError("cannot set both delimiter and delim_whitespace")
        if sep != ',':
            raise ValueError("cannot set both sep and delim_whitespace")

    # Alias sep -> delimiter.
    if delimiter is None:
        delimiter = sep

    nvtx_range_push("CUDF_READ_CSV", "purple")

    # Setup reader options
    cdef csv_reader_options args = csv_reader_options()

    if header == 'infer':
        header = -1
    header_infer = header

    if names is None:
        if header is -1:
            header_infer = 0
        if header is None:
            header_infer = -1
    else:
        if header is None:
            header_infer = -1
        for col_name in names:
            args.names.push_back(str(col_name).encode())

    if dtype is not None:
        if isinstance(dtype, abc.Mapping):
            for k, v in dtype.items():
                args.dtype.push_back(str(str(k)+":"+str(v)).encode())
        elif isinstance(dtype, abc.Iterable):
            for col_dtype in dtype:
                args.dtype.push_back(str(col_dtype).encode())
        else:
            args.dtype.push_back(str(dtype).encode())

    if usecols is not None:
        all_int = True
        # TODO Refactor to use `all_of()`
        for col in usecols:
            if not isinstance(col, int):
                all_int = False
                break
        if all_int:
            args.use_cols_indexes = usecols
        else:
            for col_name in usecols:
                args.use_cols_names.push_back(str(col_name).encode())

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

    for value in true_values or []:
        args.true_values.push_back(str(value).encode())
    for value in false_values or []:
        args.false_values.push_back(str(value).encode())
    for value in na_values or []:
        args.na_values.push_back(str(value).encode())

    if parse_dates is not None:
        if isinstance(parse_dates, abc.Mapping):
            raise TypeError("`parse_dates`: dictionaries are unsupported")
        if not isinstance(parse_dates, abc.Iterable):
            raise TypeError("`parse_dates`: non-lists are unsupported")
        for col in parse_dates:
            if isinstance(col, str):
                args.infer_date_names.push_back(str(col).encode())
            elif isinstance(col, int):
                args.infer_date_indexes.push_back(col)
            else:
                raise TypeError("`parse_dates`: Nesting is unsupported")

    args.delimiter = delimiter.encode()[0]
    args.lineterminator = lineterminator.encode()[0]
    args.quotechar = quotechar.encode()[0]
    args.quoting = _quoting_enum[quoting]
    args.doublequote = doublequote
    args.delim_whitespace = delim_whitespace
    args.skipinitialspace = skipinitialspace
    args.dayfirst = dayfirst
    args.header = header_infer
    args.mangle_dupe_cols = mangle_dupe_cols
    if compression is not None:
        args.compression = compression.encode()
    args.decimal = decimal.encode()[0]
    args.thousands = (thousands.encode() if thousands else b'\0')[0]
    args.skip_blank_lines = skip_blank_lines
    args.comment = (comment.encode() if comment else b'\0')[0]
    args.keep_default_na = keep_default_na
    args.na_filter = na_filter
    if prefix is not None:
        args.prefix = prefix.encode()

    # Create reader from source
    cdef const unsigned char[::1] buffer = view_of_buffer(filepath_or_buffer)
    cdef string filepath
    if buffer is None:
        if not os.path.isfile(filepath_or_buffer):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filepath_or_buffer
            )
        filepath = <string>str(filepath_or_buffer).encode()

    cdef unique_ptr[csv_reader] reader
    with nogil:
        if buffer is None:
            reader = unique_ptr[csv_reader](
                new csv_reader(filepath, args)
            )
        elif buffer.shape[0] != 0:
            reader = unique_ptr[csv_reader](
                new csv_reader(<char *>&buffer[0], buffer.shape[0], args)
            )
        else:
            reader = unique_ptr[csv_reader](
                new csv_reader(<char *>NULL, 0, args)
            )

    # Read data into columns
    cdef cudf_table c_out_table
    cdef size_t c_range_offset = byte_range[0] if byte_range is not None else 0
    cdef size_t c_range_size = byte_range[1] if byte_range is not None else 0
    cdef size_type c_skiprows = skiprows if skiprows is not None else 0
    cdef size_type c_skipend = skipfooter if skipfooter is not None else 0
    cdef size_type c_nrows = nrows if nrows is not None else -1
    with nogil:
        if c_range_offset !=0 or c_range_size != 0:
            c_out_table = reader.get().read_byte_range(
                c_range_offset, c_range_size
            )
        elif c_skiprows != 0 or c_skipend != 0 or c_nrows != -1:
            c_out_table = reader.get().read_rows(
                c_skiprows, c_skipend, c_nrows
            )
        else:
            c_out_table = reader.get().read()

    # Construct dataframe from columns
    cast_col_name_to_int = names is not None and isinstance(names[0], (int))
    df = table_to_dataframe(&c_out_table, int_col_names=cast_col_name_to_int)

    # Set index if the index_col parameter is passed
    if index_col is not None and index_col is not False:
        if isinstance(index_col, int):
            df = df.set_index(df.columns[index_col])
        else:
            df = df.set_index(index_col)

    nvtx_range_pop()

    return df

cpdef write_csv(
    cols,
    path=None,
    sep=",",
    na_rep="",
    columns=None,
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

    nvtx_range_push("CUDF_WRITE_CSV", "purple")

    from cudf.core.series import Series

    cdef gdf_column* c_col
    cdef cpp_csv.csv_write_arg csv_writer = cpp_csv.csv_write_arg()

    path = str(os.path.expanduser(str(path))).encode()
    csv_writer.filepath = path
    line_terminator = line_terminator.encode()
    csv_writer.line_terminator = line_terminator
    csv_writer.delimiter = sep.encode()[0]
    na_rep = na_rep.encode()
    csv_writer.na_rep = na_rep
    # Do not expose true_value and false_value until gdf_bool type
    # changes added to cpp API
    true_value = 'True'.encode()
    csv_writer.true_value = true_value
    false_value = 'False'.encode()
    csv_writer.false_value = false_value
    csv_writer.include_header = header
    # Minimum rows per chunk allowed by csvwriter is 8
    csv_writer.rows_per_chunk = rows_per_chunk if rows_per_chunk > 8 else 8

    cdef vector[gdf_column*] list_cols

    if columns is not None:
        if not isinstance(columns, list):
            raise TypeError('columns must be a list')
        for idx, col_name in enumerate(columns):
            if col_name not in cols:
                raise NameError('column {!r} does not exist in DataFrame'
                                .format(col_name))
            col = cols[col_name]
            check_gdf_compatibility(col)
            # Workaround for string columns
            if col.dtype.type == np.object_:
                c_col = column_view_from_string_column(col, col_name)
            else:
                c_col = column_view_from_column(col, col_name)
            list_cols.push_back(c_col)
    else:
        for idx, (col_name, col) in enumerate(cols.items()):
            check_gdf_compatibility(col)
            # Workaround for string columns
            if col.dtype.type == np.object_:
                c_col = column_view_from_string_column(col, col_name)
            else:
                c_col = column_view_from_column(col, col_name)
            list_cols.push_back(c_col)

    csv_writer.columns = list_cols.data()
    csv_writer.num_cols = len(columns) if columns else len(cols)

    # Call write_csv
    with nogil:
        result = cpp_csv.write_csv(&csv_writer)

    check_gdf_error(result)

    for c_col in list_cols:
        free_column(c_col)

    nvtx_range_pop()

    return None
