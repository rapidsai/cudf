# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move

import pandas as pd
import cudf
import numpy as np

from cudf._lib.cpp.types cimport size_type

import collections.abc as abc
import errno
from io import BytesIO, StringIO
import os

from enum import IntEnum

from libcpp cimport bool

from libc.stdint cimport int32_t

from cudf._lib.cpp.io.csv cimport (
    read_csv as cpp_read_csv,
    csv_reader_options,
    write_csv as cpp_write_csv,
    csv_writer_options,
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
from cudf._lib.table cimport Table, make_table_view
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


cdef csv_reader_options make_csv_reader_options(
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
    cdef compression_type c_compression
    cdef size_type c_header
    cdef string c_prefix
    cdef vector[string] c_names
    cdef size_t c_byte_range_offset = (
        byte_range[0] if byte_range is not None else 0
    )
    cdef size_t c_byte_range_size = (
        byte_range[1] if byte_range is not None else 0
    )
    cdef vector[int] c_use_cols_indexes
    cdef vector[string] c_use_cols_names
    cdef size_type c_nrows = nrows if nrows is not None else -1
    cdef quote_style c_quoting
    cdef vector[string] c_infer_date_names
    cdef vector[int] c_infer_date_indexes
    cdef vector[string] c_dtypes
    cdef vector[string] c_true_values
    cdef vector[string] c_false_values
    cdef vector[string] c_na_values

    # Reader settings
    if compression is None:
        c_compression = compression_type.NONE
    else:
        compression = str(compression)
        compression = Compression[compression.upper()]
        c_compression = <compression_type> (
            <underlying_type_t_compression> compression
        )

    if quoting == 1:
        c_quoting = quote_style.QUOTE_ALL
    elif quoting == 2:
        c_quoting = quote_style.QUOTE_NONNUMERIC
    elif quoting == 3:
        c_quoting = quote_style.QUOTE_NONE
    else:
        # Default value
        c_quoting = quote_style.QUOTE_MINIMAL

    cdef csv_reader_options csv_reader_options_c = move(
        csv_reader_options.builder(c_source_info)
        .compression(c_compression)
        .mangle_dupe_cols(mangle_dupe_cols)
        .byte_range_offset(c_byte_range_offset)
        .byte_range_size(c_byte_range_size)
        .nrows(c_nrows)
        .skiprows(skiprows)
        .skipfooter(skipfooter)
        .quoting(c_quoting)
        .lineterminator(ord(lineterminator))
        .quotechar(ord(quotechar))
        .decimal(ord(decimal))
        .delim_whitespace(delim_whitespace)
        .skipinitialspace(skipinitialspace)
        .skip_blank_lines(skip_blank_lines)
        .doublequote(doublequote)
        .keep_default_na(keep_default_na)
        .na_filter(na_filter)
        .dayfirst(dayfirst)
        .build()
    )

    if names is not None:
        # explicitly mentioned name, so don't check header
        if header is None or header == 'infer':
            csv_reader_options_c.set_header(-1)
        else:
            csv_reader_options_c.set_header(header)

        c_names.reserve(len(names))
        for name in names:
            c_names.push_back(str(name).encode())
        csv_reader_options_c.set_names(c_names)
    else:
        if header is None:
            csv_reader_options_c.set_header(-1)
        elif header == 'infer':
            csv_reader_options_c.set_header(0)
        else:
            csv_reader_options_c.set_header(header)

    if prefix is not None:
        csv_reader_options_c.set_prefix(prefix.encode())

    if usecols is not None:
        all_int = all(isinstance(col, int) for col in usecols)
        if all_int:
            c_use_cols_indexes.reserve(len(usecols))
            c_use_cols_indexes = usecols
            csv_reader_options_c.set_use_cols_indexes(c_use_cols_indexes)
        else:
            c_use_cols_names.reserve(len(usecols))
            for col_name in usecols:
                c_use_cols_names.push_back(
                    str(col_name).encode()
                )
            csv_reader_options_c.set_use_cols_names(c_use_cols_names)

    if delimiter is not None:
        csv_reader_options_c.set_delimiter(ord(delimiter))

    if thousands is not None:
        csv_reader_options_c.set_thousands(ord(thousands))

    if comment is not None:
        csv_reader_options_c.set_comment(ord(comment))

    if parse_dates is not None:
        if isinstance(parse_dates, abc.Mapping):
            raise NotImplementedError(
                "`parse_dates`: dictionaries are unsupported")
        if not isinstance(parse_dates, abc.Iterable):
            raise NotImplementedError(
                "`parse_dates`: non-lists are unsupported")
        for col in parse_dates:
            if isinstance(col, str):
                c_infer_date_names.push_back(str(col).encode())
            elif isinstance(col, int):
                c_infer_date_indexes.push_back(col)
            else:
                raise NotImplementedError(
                    "`parse_dates`: Nesting is unsupported")
        csv_reader_options_c.set_infer_date_names(c_infer_date_names)
        csv_reader_options_c.set_infer_date_indexes(c_infer_date_indexes)

    if dtype is not None:
        if isinstance(dtype, abc.Mapping):
            c_dtypes.reserve(len(dtype))
            for k, v in dtype.items():
                c_dtypes.push_back(
                    str(
                        str(k)+":"+
                        _get_cudf_compatible_str_from_dtype(v)
                    ).encode()
                )
        elif (
            cudf.utils.dtypes.is_scalar(dtype) or
            isinstance(dtype, (
                np.dtype, pd.core.dtypes.dtypes.ExtensionDtype, type
            ))
        ):
            c_dtypes.reserve(1)
            c_dtypes.push_back(
                _get_cudf_compatible_str_from_dtype(dtype).encode()
            )
        elif isinstance(dtype, abc.Iterable):
            c_dtypes.reserve(len(dtype))
            for col_dtype in dtype:
                c_dtypes.push_back(
                    _get_cudf_compatible_str_from_dtype(col_dtype).encode()
                )
        else:
            raise ValueError(
                "dtype should be a scalar/str/list-like/dict-like"
            )

        csv_reader_options_c.set_dtypes(c_dtypes)

    if true_values is not None:
        c_true_values.reserve(len(true_values))
        for tv in true_values:
            c_true_values.push_back(tv.encode())
        csv_reader_options_c.set_true_values(c_true_values)

    if false_values is not None:
        c_false_values.reserve(len(false_values))
        for fv in c_false_values:
            c_false_values.push_back(fv.encode())
        csv_reader_options_c.set_false_values(c_false_values)

    if na_values is not None:
        c_na_values.reserve(len(na_values))
        for nv in na_values:
            c_na_values.push_back(nv.encode())
        csv_reader_options_c.set_na_values(c_na_values)

    return csv_reader_options_c


def validate_args(
    object delimiter,
    object sep,
    bool delim_whitespace,
    object decimal,
    object thousands,
    object nrows,
    int skipfooter,
    object byte_range,
    int skiprows
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
    object datasource,
    object lineterminator="\n",
    object quotechar='"',
    int quoting=0,
    bool doublequote=True,
    object header="infer",
    bool mangle_dupe_cols=True,
    object usecols=None,
    object sep=",",
    object delimiter=None,
    bool delim_whitespace=False,
    bool skipinitialspace=False,
    object names=None,
    object dtype=None,
    int skipfooter=0,
    int skiprows=0,
    bool dayfirst=False,
    object compression="infer",
    object thousands=None,
    object decimal=".",
    object true_values=None,
    object false_values=None,
    object nrows=None,
    object byte_range=None,
    bool skip_blank_lines=True,
    object parse_dates=None,
    object comment=None,
    object na_values=None,
    bool keep_default_na=True,
    bool na_filter=True,
    object prefix=None,
    object index_col=None,
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

    cdef csv_reader_options read_csv_options_c = make_csv_reader_options(
        datasource, lineterminator, quotechar, quoting, doublequote,
        header, mangle_dupe_cols, usecols, delimiter, delim_whitespace,
        skipinitialspace, names, dtype, skipfooter, skiprows, dayfirst,
        compression, thousands, decimal, true_values, false_values, nrows,
        byte_range, skip_blank_lines, parse_dates, comment, na_values,
        keep_default_na, na_filter, prefix, index_col)

    cdef table_with_metadata c_result
    with nogil:
        c_result = move(cpp_read_csv(read_csv_options_c))

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
    object path_or_buf=None,
    object sep=",",
    object na_rep="",
    bool header=True,
    object line_terminator="\n",
    int rows_per_chunk=8,
    bool index=True,
):
    """
    Cython function to call into libcudf API, see `write_csv`.

    See Also
    --------
    cudf.io.csv.to_csv
    """

    cdef table_view input_table_view = \
        table.view() if index is True else table.data_view()
    cdef bool include_header_c = header
    cdef char delim_c = ord(sep)
    cdef string line_term_c = line_terminator.encode()
    cdef string na_c = na_rep.encode()
    cdef int rows_per_chunk_c = rows_per_chunk
    cdef table_metadata metadata_ = table_metadata()
    cdef string true_value_c = 'True'.encode()
    cdef string false_value_c = 'False'.encode()
    cdef unique_ptr[data_sink] data_sink_c
    cdef sink_info sink_info_c = make_sink_info(path_or_buf, data_sink_c)

    if header is True:
        all_names = columns_apply_na_rep(table._column_names, na_rep)
        if index is True:
            all_names = table._index.names + all_names

        if len(all_names) > 0:
            metadata_.column_names.reserve(len(all_names))
            if len(all_names) == 1:
                if all_names[0] in (None, ''):
                    metadata_.column_names.push_back('""'.encode())
                else:
                    metadata_.column_names.push_back(
                        str(all_names[0]).encode()
                    )
            else:
                for idx, col_name in enumerate(all_names):
                    if col_name is None:
                        metadata_.column_names.push_back(''.encode())
                    else:
                        metadata_.column_names.push_back(
                            str(col_name).encode()
                        )

    cdef csv_writer_options options = move(
        csv_writer_options.builder(sink_info_c, input_table_view)
        .metadata(&metadata_)
        .na_rep(na_c)
        .include_header(include_header_c)
        .rows_per_chunk(rows_per_chunk_c)
        .line_terminator(line_term_c)
        .inter_column_delimiter(delim_c)
        .true_value(true_value_c)
        .false_value(false_value_c)
        .build()
    )

    with nogil:
        cpp_write_csv(options)


def _get_cudf_compatible_str_from_dtype(dtype):
    # TODO: Remove this Error message once the
    # following issue is fixed:
    # https://github.com/rapidsai/cudf/issues/3960
    if cudf.utils.dtypes.is_categorical_dtype(dtype):
        raise NotImplementedError(
            "CategoricalDtype as dtype is not yet "
            "supported in CSV reader"
        )

    if (
        str(dtype) in cudf.utils.dtypes.ALL_TYPES or
        str(dtype) in {
            "hex", "hex32", "hex64", "date", "date32", "timestamp",
            "timestamp[us]", "timestamp[s]", "timestamp[ms]", "timestamp[ns]",
            "date64"
        }
    ):
        return str(dtype)
    pd_dtype = pd.core.dtypes.common.pandas_dtype(dtype)

    if pd_dtype in cudf.utils.dtypes.pandas_dtypes_to_cudf_dtypes:
        return str(cudf.utils.dtypes.pandas_dtypes_to_cudf_dtypes[pd_dtype])
    elif isinstance(pd_dtype, np.dtype) and pd_dtype.kind in ("O", "U"):
        return "str"
    elif (
        pd_dtype in cudf.utils.dtypes.cudf_dtypes_to_pandas_dtypes or
        str(pd_dtype) in cudf.utils.dtypes.ALL_TYPES or
        cudf.utils.dtypes.is_categorical_dtype(pd_dtype)
    ):
        return str(pd_dtype)
    else:
        raise ValueError(f"dtype not understood: {dtype}")


def columns_apply_na_rep(column_names, na_rep):
    return tuple(
        na_rep if pd.isnull(col_name)
        else col_name
        for col_name in column_names
    )
