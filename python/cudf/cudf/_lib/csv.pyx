# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool

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

from cudf._lib.utils cimport data_from_pylibcudf_io

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
    options = (
        plc.io.csv.CsvReaderOptions.builder(plc.io.SourceInfo([datasource]))
        .compression(c_compression)
        .mangle_dupe_cols(mangle_dupe_cols)
        .byte_range_offset(byte_range[0])
        .byte_range_size(byte_range[1])
        .nrows(nrows if nrows is not None else -1)
        .skiprows(skiprows)
        .skipfooter(skipfooter)
        .quoting(quoting)
        .lineterminator(str(lineterminator))
        .quotechar(quotechar)
        .decimal(decimal)
        .delim_whitespace(delim_whitespace)
        .skipinitialspace(skipinitialspace)
        .skip_blank_lines(skip_blank_lines)
        .doublequote(doublequote)
        .keep_default_na(keep_default_na)
        .na_filter(na_filter)
        .dayfirst(dayfirst)
        .build()
    )

    options.set_header(header)

    if names is not None:
        options.set_names([str(name) for name in names])

    if prefix is not None:
        options.set_prefix(prefix)

    if usecols is not None:
        if all(isinstance(col, int) for col in usecols):
            options.set_use_cols_indexes(list(usecols))
        else:
            options.set_use_cols_names([str(name) for name in usecols])

    if delimiter is not None:
        options.set_delimiter(delimiter)

    if thousands is not None:
        options.set_thousands(thousands)

    if comment is not None:
        options.set_comment(comment)

    if parse_dates is not None:
        options.set_parse_dates(list(parse_dates))

    if hex_cols is not None:
        options.set_parse_hex(list(hex_cols))

    options.set_dtypes(new_dtypes)

    if true_values is not None:
        options.set_true_values([str(val) for val in true_values])

    if false_values is not None:
        options.set_false_values([str(val) for val in false_values])

    if na_values is not None:
        options.set_na_values([str(val) for val in na_values])

    df = cudf.DataFrame._from_data(
        *data_from_pylibcudf_io(plc.io.csv.read_csv(options))
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
    index_and_not_empty = index is True and table.index is not None
    columns = [
        col.to_pylibcudf(mode="read") for col in table.index._columns
    ] if index_and_not_empty else []
    columns.extend(col.to_pylibcudf(mode="read") for col in table._columns)
    col_names = []
    if header:
        all_names = list(table.index.names) if index_and_not_empty else []
        all_names.extend(
            na_rep if name is None or pd.isnull(name)
            else name for name in table._column_names
        )
        col_names = [
            '""' if (name in (None, '') and len(all_names) == 1)
            else (str(name) if name not in (None, '') else '')
            for name in all_names
        ]
    try:
        plc.io.csv.write_csv(
            (
                plc.io.csv.CsvWriterOptions.builder(
                    plc.io.SinkInfo([path_or_buf]), plc.Table(columns)
                )
                .names(col_names)
                .na_rep(na_rep)
                .include_header(header)
                .rows_per_chunk(rows_per_chunk)
                .line_terminator(str(lineterminator))
                .inter_column_delimiter(str(sep))
                .true_value("True")
                .false_value("False")
                .build()
            )
        )
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
