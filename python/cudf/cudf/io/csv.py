# Copyright (c) 2018-2024, NVIDIA CORPORATION.
from __future__ import annotations

import errno
import itertools
import os
import warnings
from collections import abc
from io import BytesIO, StringIO
from typing import cast

import numpy as np
import pandas as pd

import pylibcudf as plc

import cudf
from cudf._lib.column import Column
from cudf._lib.types import dtype_to_pylibcudf_type
from cudf.api.types import is_hashable, is_scalar
from cudf.core.buffer import acquire_spill_lock
from cudf.utils import ioutils
from cudf.utils.dtypes import _maybe_convert_to_default_type
from cudf.utils.performance_tracking import _performance_tracking

_CSV_HEX_TYPE_MAP = {
    "hex": np.dtype("int64"),
    "hex64": np.dtype("int64"),
    "hex32": np.dtype("int32"),
}


@_performance_tracking
@ioutils.doc_read_csv()
def read_csv(
    filepath_or_buffer,
    sep: str = ",",
    delimiter: str | None = None,
    header="infer",
    names=None,
    index_col=None,
    usecols=None,
    prefix=None,
    mangle_dupe_cols: bool = True,
    dtype=None,
    true_values=None,
    false_values=None,
    skipinitialspace: bool = False,
    skiprows: int = 0,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values=None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates=None,
    dayfirst: bool = False,
    compression="infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str = "\n",
    quotechar: str = '"',
    quoting: int = 0,
    doublequote: bool = True,
    comment: str | None = None,
    delim_whitespace: bool = False,
    byte_range: list[int] | tuple[int, int] | None = None,
    storage_options=None,
    bytes_per_thread: int | None = None,
) -> cudf.DataFrame:
    """{docstring}"""

    if delim_whitespace is not False:
        warnings.warn(
            "The 'delim_whitespace' keyword in pd.read_csv is deprecated and "
            "will be removed in a future version. Use ``sep='\\s+'`` instead",
            FutureWarning,
        )

    if bytes_per_thread is None:
        bytes_per_thread = ioutils._BYTES_PER_THREAD_DEFAULT

    filepath_or_buffer = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        iotypes=(BytesIO, StringIO),
        storage_options=storage_options,
        bytes_per_thread=bytes_per_thread,
    )
    filepath_or_buffer = ioutils._select_single_source(
        filepath_or_buffer, "read_csv"
    )

    if na_values is not None and is_scalar(na_values):
        na_values = [na_values]

    if not isinstance(filepath_or_buffer, (BytesIO, StringIO, bytes)):
        if not os.path.isfile(filepath_or_buffer):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filepath_or_buffer
            )

    if isinstance(filepath_or_buffer, StringIO):
        filepath_or_buffer = filepath_or_buffer.read().encode()
    elif isinstance(filepath_or_buffer, str) and not os.path.isfile(
        filepath_or_buffer
    ):
        filepath_or_buffer = filepath_or_buffer.encode()

    _validate_args(
        delimiter,
        sep,
        delim_whitespace,
        decimal,
        thousands,
        nrows,
        skipfooter,
        byte_range,
        skiprows,
    )

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
        if header is None or header == "infer":
            header = -1
        else:
            header = header
        names = list(names)
    else:
        if header is None:
            header = -1
        elif header == "infer":
            header = 0

    hex_cols: list[abc.Hashable] = []
    new_dtypes: list[plc.DataType] | dict[abc.Hashable, plc.DataType] = []
    if dtype is not None:
        if isinstance(dtype, abc.Mapping):
            new_dtypes = {}
            for k, col_type in dtype.items():
                if is_hashable(col_type) and col_type in _CSV_HEX_TYPE_MAP:
                    col_type = _CSV_HEX_TYPE_MAP[col_type]
                    hex_cols.append(str(k))

                new_dtypes[k] = _get_plc_data_type_from_dtype(
                    cudf.dtype(col_type)
                )
        elif cudf.api.types.is_scalar(dtype) or isinstance(
            dtype, (np.dtype, pd.api.extensions.ExtensionDtype, type)
        ):
            if is_hashable(dtype) and dtype in _CSV_HEX_TYPE_MAP:
                dtype = _CSV_HEX_TYPE_MAP[dtype]
                hex_cols.append(0)

            cast(list, new_dtypes).append(_get_plc_data_type_from_dtype(dtype))
        elif isinstance(dtype, abc.Collection):
            for index, col_dtype in enumerate(dtype):
                if is_hashable(col_dtype) and col_dtype in _CSV_HEX_TYPE_MAP:
                    col_dtype = _CSV_HEX_TYPE_MAP[col_dtype]
                    hex_cols.append(index)

                new_dtypes.append(_get_plc_data_type_from_dtype(col_dtype))
        else:
            raise ValueError(
                "dtype should be a scalar/str/list-like/dict-like"
            )
    options = (
        plc.io.csv.CsvReaderOptions.builder(
            plc.io.SourceInfo([filepath_or_buffer])
        )
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

    table_w_meta = plc.io.csv.read_csv(options)
    data = {
        name: Column.from_pylibcudf(col)
        for name, col in zip(
            table_w_meta.column_names(include_children=False),
            table_w_meta.columns,
            strict=True,
        )
    }

    df = cudf.DataFrame._from_data(data)

    if isinstance(dtype, abc.Mapping):
        for k, v in dtype.items():
            if isinstance(cudf.dtype(v), cudf.CategoricalDtype):
                df._data[str(k)] = df._data[str(k)].astype(v)
    elif dtype == "category" or isinstance(dtype, cudf.CategoricalDtype):
        df = df.astype(dtype)
    elif isinstance(dtype, abc.Collection) and not is_scalar(dtype):
        for index, col_dtype in enumerate(dtype):
            if isinstance(cudf.dtype(col_dtype), cudf.CategoricalDtype):
                col_name = df._column_names[index]
                df._data[col_name] = df._data[col_name].astype(col_dtype)

    if names is not None and len(names) and isinstance(names[0], int):
        df.columns = [int(x) for x in df._data]
    elif (
        names is None
        and header == -1
        and cudf.get_option("mode.pandas_compatible")
    ):
        df.columns = [int(x) for x in df._column_names]

    # Set index if the index_col parameter is passed
    if index_col is not None and index_col is not False:
        if isinstance(index_col, int):
            index_col_name = df._data.get_labels_by_index(index_col)[0]
            df = df.set_index(index_col_name)
            if (
                isinstance(index_col_name, str)
                and names is None
                and orig_header == "infer"
            ):
                if index_col_name.startswith("Unnamed:"):
                    # TODO: Try to upstream it to libcudf
                    # csv reader in future
                    df.index.name = None
            elif names is None:
                df.index.name = index_col
        else:
            df = df.set_index(index_col)

    if dtype is None or isinstance(dtype, abc.Mapping):
        # There exists some dtypes in the result columns that is inferred.
        # Find them and map them to the default dtypes.
        specified_dtypes = {} if dtype is None else dtype
        default_dtypes = {}
        for name, dt in df._dtypes:
            if name in specified_dtypes:
                continue
            elif dt == np.dtype("i1"):
                # csv reader reads all null column as int8.
                # The dtype should remain int8.
                default_dtypes[name] = dt
            else:
                default_dtypes[name] = _maybe_convert_to_default_type(dt)

        if default_dtypes:
            df = df.astype(default_dtypes)

    return df


@_performance_tracking
@ioutils.doc_to_csv()
def to_csv(
    df: cudf.DataFrame,
    path_or_buf=None,
    sep: str = ",",
    na_rep: str = "",
    columns=None,
    header: bool = True,
    index: bool = True,
    encoding=None,
    compression=None,
    lineterminator: str = "\n",
    chunksize: int | None = None,
    storage_options=None,
):
    """{docstring}"""

    if not isinstance(sep, str):
        raise TypeError(f'"sep" must be string, not {type(sep).__name__}')
    elif len(sep) > 1:
        raise TypeError('"sep" must be a 1-character string')

    if encoding and encoding != "utf-8":
        error_msg = (
            f"Encoding {encoding} is not supported. "
            + "Currently, only utf-8 encoding is supported."
        )
        raise NotImplementedError(error_msg)

    if compression:
        error_msg = "Writing compressed csv is not currently supported in cudf"
        raise NotImplementedError(error_msg)

    return_as_string = False
    if path_or_buf is None:
        path_or_buf = StringIO()
        return_as_string = True

    path_or_buf = ioutils.get_writer_filepath_or_buffer(
        path_or_data=path_or_buf, mode="w", storage_options=storage_options
    )

    if columns is not None:
        try:
            df = df[columns]
        except KeyError:
            raise NameError(
                "Dataframe doesn't have the labels provided in columns"
            )

    for _, dtype in df._dtypes:
        if isinstance(dtype, (cudf.ListDtype, cudf.StructDtype)):
            raise NotImplementedError(
                "Writing to csv format is not yet supported with "
                f"{dtype} columns."
            )

    # TODO: Need to typecast categorical columns to the underlying
    # categories dtype to write the actual data to csv. Remove this
    # workaround once following issue is fixed:
    # https://github.com/rapidsai/cudf/issues/6661
    if any(
        isinstance(dtype, cudf.CategoricalDtype) for _, dtype in df._dtypes
    ) or isinstance(df.index, cudf.CategoricalIndex):
        df = df.copy(deep=False)
        for col_name, col in df._column_labels_and_values:
            if isinstance(col.dtype, cudf.CategoricalDtype):
                df._data[col_name] = col.astype(col.dtype.categories.dtype)

        if isinstance(df.index, cudf.CategoricalIndex):
            df.index = df.index.astype(df.index.categories.dtype)

    rows_per_chunk = chunksize if chunksize else len(df)

    if ioutils.is_fsspec_open_file(path_or_buf):
        with path_or_buf as file_obj:
            file_obj = ioutils.get_IOBase_writer(file_obj)
            _plc_write_csv(
                df,
                path_or_buf=file_obj,
                sep=sep,
                na_rep=na_rep,
                header=header,
                lineterminator=lineterminator,
                rows_per_chunk=rows_per_chunk,
                index=index,
            )
    else:
        _plc_write_csv(
            df,
            path_or_buf=path_or_buf,
            sep=sep,
            na_rep=na_rep,
            header=header,
            lineterminator=lineterminator,
            rows_per_chunk=rows_per_chunk,
            index=index,
        )

    if return_as_string:
        path_or_buf.seek(0)
        return path_or_buf.read()


@acquire_spill_lock()
def _plc_write_csv(
    table: cudf.DataFrame,
    path_or_buf=None,
    sep: str = ",",
    na_rep: str = "",
    header: bool = True,
    lineterminator: str = "\n",
    rows_per_chunk: int = 8,
    index: bool = True,
) -> None:
    iter_columns = (
        itertools.chain(table.index._columns, table._columns)
        if index
        else table._columns
    )
    columns = [col.to_pylibcudf(mode="read") for col in iter_columns]
    col_names = []
    if header:
        table_names = (
            na_rep if name is None or pd.isnull(name) else name
            for name in table._column_names
        )
        iter_names = (
            itertools.chain(table.index.names, table_names)
            if index
            else table_names
        )
        all_names = list(iter_names)
        col_names = [
            '""'
            if (name in (None, "") and len(all_names) == 1)
            else (str(name) if name not in (None, "") else "")
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
    except OverflowError as err:
        raise OverflowError(
            f"Writing CSV file with chunksize={rows_per_chunk} failed. "
            "Consider providing a smaller chunksize argument."
        ) from err


def _validate_args(
    delimiter: str | None,
    sep: str,
    delim_whitespace: bool,
    decimal: str,
    thousands: str | None,
    nrows: int | None,
    skipfooter: int,
    byte_range: list[int] | tuple[int, int] | None,
    skiprows: int,
) -> None:
    if delim_whitespace:
        if delimiter is not None:
            raise ValueError("cannot set both delimiter and delim_whitespace")
        if sep != ",":
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
            raise ValueError(
                "cannot manually limit rows to be read when using the byte range parameter"
            )


def _get_plc_data_type_from_dtype(dtype) -> plc.DataType:
    # TODO: Remove this work-around Dictionary types
    # in libcudf are fully mapped to categorical columns:
    # https://github.com/rapidsai/cudf/issues/3960
    if isinstance(dtype, cudf.CategoricalDtype):
        dtype = dtype.categories.dtype
    elif dtype == "category":
        dtype = "str"

    if isinstance(dtype, str):
        if dtype == "date32":
            return plc.DataType(plc.types.TypeId.TIMESTAMP_DAYS)
        elif dtype in ("date", "date64"):
            return plc.DataType(plc.types.TypeId.TIMESTAMP_MILLISECONDS)
        elif dtype == "timestamp":
            return plc.DataType(plc.types.TypeId.TIMESTAMP_MILLISECONDS)
        elif dtype == "timestamp[us]":
            return plc.DataType(plc.types.TypeId.TIMESTAMP_MICROSECONDS)
        elif dtype == "timestamp[s]":
            return plc.DataType(plc.types.TypeId.TIMESTAMP_SECONDS)
        elif dtype == "timestamp[ms]":
            return plc.DataType(plc.types.TypeId.TIMESTAMP_MILLISECONDS)
        elif dtype == "timestamp[ns]":
            return plc.DataType(plc.types.TypeId.TIMESTAMP_NANOSECONDS)

    dtype = cudf.dtype(dtype)
    return dtype_to_pylibcudf_type(dtype)
