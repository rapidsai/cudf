# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

cimport pylibcudf.libcudf.types as libcudf_types

from cudf._lib.types cimport dtype_to_pylibcudf_type

import errno
import os
from collections import abc
from io import BytesIO, StringIO

import numpy as np
import pandas as pd

import cudf
from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool

from pylibcudf.libcudf.io.csv cimport (
    csv_writer_options,
    write_csv as cpp_write_csv,
)
from pylibcudf.libcudf.io.data_sink cimport data_sink
from pylibcudf.libcudf.io.types cimport sink_info
from pylibcudf.libcudf.table.table_view cimport table_view

from cudf._lib.io.utils cimport make_sink_info
from cudf._lib.utils cimport data_from_pylibcudf_io, table_view_from_table

import pylibcudf as plc

from cudf.api.types import is_hashable

from pylibcudf.types cimport DataType

CSV_HEX_TYPE_MAP = {
    "hex": np.dtype("int64"),
    "hex64": np.dtype("int64"),
    "hex32": np.dtype("int32")
}


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
):
    """
    Cython function to call into libcudf API, see `read_csv`.

    See Also
    --------
    cudf.read_csv
    """

    if not isinstance(datasource, (BytesIO, StringIO, bytes)):
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

    delimiter = str(delimiter)

    if byte_range is None:
        byte_range = (0, 0)

    if compression is None:
        c_compression = plc.io.types.CompressionType.NONE
    else:
        compression_map = {
            "infer": plc.io.types.CompressionType.AUTO,
            "gzip": plc.io.types.CompressionType.GZIP,
            "bz2": plc.io.types.CompressionType.BZIP2,
            "zip": plc.io.types.CompressionType.ZIP,
        }
        c_compression = compression_map[compression]

    # We need this later when setting index cols
    orig_header = header

    if names is not None:
        # explicitly mentioned name, so don't check header
        if header is None or header == 'infer':
            header = -1
        else:
            header = header
        names = list(names)
    else:
        if header is None:
            header = -1
        elif header == 'infer':
            header = 0

    hex_cols = []

    new_dtypes = []
    if dtype is not None:
        if isinstance(dtype, abc.Mapping):
            new_dtypes = dict()
            for k, v in dtype.items():
                col_type = v
                if is_hashable(v) and v in CSV_HEX_TYPE_MAP:
                    col_type = CSV_HEX_TYPE_MAP[v]
                    hex_cols.append(str(k))

                new_dtypes[k] = _get_plc_data_type_from_dtype(
                    cudf.dtype(col_type)
                )
        elif (
            cudf.api.types.is_scalar(dtype) or
            isinstance(dtype, (
                np.dtype, pd.api.extensions.ExtensionDtype, type
            ))
        ):
            if is_hashable(dtype) and dtype in CSV_HEX_TYPE_MAP:
                dtype = CSV_HEX_TYPE_MAP[dtype]
                hex_cols.append(0)

            new_dtypes.append(
                _get_plc_data_type_from_dtype(dtype)
            )
        elif isinstance(dtype, abc.Collection):
            for index, col_dtype in enumerate(dtype):
                if is_hashable(col_dtype) and col_dtype in CSV_HEX_TYPE_MAP:
                    col_dtype = CSV_HEX_TYPE_MAP[col_dtype]
                    hex_cols.append(index)

                new_dtypes.append(
                    _get_plc_data_type_from_dtype(col_dtype)
                )
        else:
            raise ValueError(
                "dtype should be a scalar/str/list-like/dict-like"
            )

    lineterminator = str(lineterminator)

    df = cudf.DataFrame._from_data(
        *data_from_pylibcudf_io(
            plc.io.csv.read_csv(
                plc.io.SourceInfo([datasource]),
                lineterminator=lineterminator,
                quotechar = quotechar,
                quoting = quoting,
                doublequote = doublequote,
                header = header,
                mangle_dupe_cols = mangle_dupe_cols,
                usecols = usecols,
                delimiter = delimiter,
                delim_whitespace = delim_whitespace,
                skipinitialspace = skipinitialspace,
                col_names = names,
                dtypes = new_dtypes,
                skipfooter = skipfooter,
                skiprows = skiprows,
                dayfirst = dayfirst,
                compression = c_compression,
                thousands = thousands,
                decimal = decimal,
                true_values = true_values,
                false_values = false_values,
                nrows = nrows if nrows is not None else -1,
                byte_range_offset = byte_range[0],
                byte_range_size = byte_range[1],
                skip_blank_lines = skip_blank_lines,
                parse_dates = parse_dates,
                parse_hex = hex_cols,
                comment = comment,
                na_values = na_values,
                keep_default_na = keep_default_na,
                na_filter = na_filter,
                prefix = prefix,
            )
        )
    )

    if dtype is not None:
        if isinstance(dtype, abc.Mapping):
            for k, v in dtype.items():
                if isinstance(cudf.dtype(v), cudf.CategoricalDtype):
                    df._data[str(k)] = df._data[str(k)].astype(v)
        elif (
            cudf.api.types.is_scalar(dtype) or
            isinstance(dtype, (
                np.dtype, pd.api.extensions.ExtensionDtype, type
            ))
        ):
            if isinstance(cudf.dtype(dtype), cudf.CategoricalDtype):
                df = df.astype(dtype)
        elif isinstance(dtype, abc.Collection):
            for index, col_dtype in enumerate(dtype):
                if isinstance(cudf.dtype(col_dtype), cudf.CategoricalDtype):
                    col_name = df._column_names[index]
                    df._data[col_name] = df._data[col_name].astype(col_dtype)

    if names is not None and len(names) and isinstance(names[0], int):
        df.columns = [int(x) for x in df._data]
    elif names is None and header == -1 and cudf.get_option("mode.pandas_compatible"):
        df.columns = [int(x) for x in df._column_names]

    # Set index if the index_col parameter is passed
    if index_col is not None and index_col is not False:
        if isinstance(index_col, int):
            index_col_name = df._data.get_labels_by_index(index_col)[0]
            df = df.set_index(index_col_name)
            if isinstance(index_col_name, str) and \
                    names is None and orig_header == "infer":
                if index_col_name.startswith("Unnamed:"):
                    # TODO: Try to upstream it to libcudf
                    # csv reader in future
                    df._index.name = None
            elif names is None:
                df._index.name = index_col
        else:
            df = df.set_index(index_col)

    return df


@acquire_spill_lock()
def write_csv(
    table,
    object path_or_buf=None,
    object sep=",",
    object na_rep="",
    bool header=True,
    object lineterminator="\n",
    int rows_per_chunk=8,
    bool index=True,
):
    """
    Cython function to call into libcudf API, see `write_csv`.

    See Also
    --------
    cudf.to_csv
    """
    cdef table_view input_table_view = table_view_from_table(
        table, not index
    )
    cdef bool include_header_c = header
    cdef char delim_c = ord(sep)
    cdef string line_term_c = lineterminator.encode()
    cdef string na_c = na_rep.encode()
    cdef int rows_per_chunk_c = rows_per_chunk
    cdef vector[string] col_names
    cdef string true_value_c = 'True'.encode()
    cdef string false_value_c = 'False'.encode()
    cdef unique_ptr[data_sink] data_sink_c
    cdef sink_info sink_info_c = make_sink_info(path_or_buf, data_sink_c)

    if header is True:
        all_names = columns_apply_na_rep(table._column_names, na_rep)
        if index is True:
            all_names = table._index.names + all_names

        if len(all_names) > 0:
            col_names.reserve(len(all_names))
            if len(all_names) == 1:
                if all_names[0] in (None, ''):
                    col_names.push_back('""'.encode())
                else:
                    col_names.push_back(
                        str(all_names[0]).encode()
                    )
            else:
                for idx, col_name in enumerate(all_names):
                    if col_name is None:
                        col_names.push_back(''.encode())
                    else:
                        col_names.push_back(
                            str(col_name).encode()
                        )

    cdef csv_writer_options options = move(
        csv_writer_options.builder(sink_info_c, input_table_view)
        .names(col_names)
        .na_rep(na_c)
        .include_header(include_header_c)
        .rows_per_chunk(rows_per_chunk_c)
        .line_terminator(line_term_c)
        .inter_column_delimiter(delim_c)
        .true_value(true_value_c)
        .false_value(false_value_c)
        .build()
    )

    try:
        with nogil:
            cpp_write_csv(options)
    except OverflowError:
        raise OverflowError(
            f"Writing CSV file with chunksize={rows_per_chunk} failed. "
            "Consider providing a smaller chunksize argument."
        )


cdef DataType _get_plc_data_type_from_dtype(object dtype) except *:
    # TODO: Remove this work-around Dictionary types
    # in libcudf are fully mapped to categorical columns:
    # https://github.com/rapidsai/cudf/issues/3960
    if isinstance(dtype, cudf.CategoricalDtype):
        dtype = dtype.categories.dtype
    elif dtype == "category":
        dtype = "str"

    if isinstance(dtype, str):
        if str(dtype) == "date32":
            return DataType(
                libcudf_types.type_id.TIMESTAMP_DAYS
            )
        elif str(dtype) in ("date", "date64"):
            return DataType(
                libcudf_types.type_id.TIMESTAMP_MILLISECONDS
            )
        elif str(dtype) == "timestamp":
            return DataType(
                libcudf_types.type_id.TIMESTAMP_MILLISECONDS
            )
        elif str(dtype) == "timestamp[us]":
            return DataType(
                libcudf_types.type_id.TIMESTAMP_MICROSECONDS
            )
        elif str(dtype) == "timestamp[s]":
            return DataType(
                libcudf_types.type_id.TIMESTAMP_SECONDS
            )
        elif str(dtype) == "timestamp[ms]":
            return DataType(
                libcudf_types.type_id.TIMESTAMP_MILLISECONDS
            )
        elif str(dtype) == "timestamp[ns]":
            return DataType(
                libcudf_types.type_id.TIMESTAMP_NANOSECONDS
            )

    dtype = cudf.dtype(dtype)
    return dtype_to_pylibcudf_type(dtype)


def columns_apply_na_rep(column_names, na_rep):
    return tuple(
        na_rep if pd.isnull(col_name)
        else col_name
        for col_name in column_names
    )
