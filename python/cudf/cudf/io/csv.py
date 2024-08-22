# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import warnings
from collections import abc
from io import BytesIO, StringIO

import numpy as np

import cudf
from cudf import _lib as libcudf
from cudf.api.types import is_scalar
from cudf.utils import ioutils
from cudf.utils.dtypes import _maybe_convert_to_default_type
from cudf.utils.performance_tracking import _performance_tracking


@_performance_tracking
@ioutils.doc_read_csv()
def read_csv(
    filepath_or_buffer,
    sep=",",
    delimiter=None,
    header="infer",
    names=None,
    index_col=None,
    usecols=None,
    prefix=None,
    mangle_dupe_cols=True,
    dtype=None,
    true_values=None,
    false_values=None,
    skipinitialspace=False,
    skiprows=0,
    skipfooter=0,
    nrows=None,
    na_values=None,
    keep_default_na=True,
    na_filter=True,
    skip_blank_lines=True,
    parse_dates=None,
    dayfirst=False,
    compression="infer",
    thousands=None,
    decimal=".",
    lineterminator="\n",
    quotechar='"',
    quoting=0,
    doublequote=True,
    comment=None,
    delim_whitespace=False,
    byte_range=None,
    storage_options=None,
    bytes_per_thread=None,
):
    """{docstring}"""

    if delim_whitespace is not False:
        warnings.warn(
            "The 'delim_whitespace' keyword in pd.read_csv is deprecated and "
            "will be removed in a future version. Use ``sep='\\s+'`` instead",
            FutureWarning,
        )

    if bytes_per_thread is None:
        bytes_per_thread = ioutils._BYTES_PER_THREAD_DEFAULT

    is_single_filepath_or_buffer = ioutils.ensure_single_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        storage_options=storage_options,
    )
    if not is_single_filepath_or_buffer:
        raise NotImplementedError(
            "`read_csv` does not yet support reading multiple files"
        )

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

    df = libcudf.csv.read_csv(
        filepath_or_buffer,
        lineterminator=lineterminator,
        quotechar=quotechar,
        quoting=quoting,
        doublequote=doublequote,
        header=header,
        mangle_dupe_cols=mangle_dupe_cols,
        usecols=usecols,
        sep=sep,
        delimiter=delimiter,
        delim_whitespace=delim_whitespace,
        skipinitialspace=skipinitialspace,
        names=names,
        dtype=dtype,
        skipfooter=skipfooter,
        skiprows=skiprows,
        dayfirst=dayfirst,
        compression=compression,
        thousands=thousands,
        decimal=decimal,
        true_values=true_values,
        false_values=false_values,
        nrows=nrows,
        byte_range=byte_range,
        skip_blank_lines=skip_blank_lines,
        parse_dates=parse_dates,
        comment=comment,
        na_values=na_values,
        keep_default_na=keep_default_na,
        na_filter=na_filter,
        prefix=prefix,
        index_col=index_col,
    )

    if dtype is None or isinstance(dtype, abc.Mapping):
        # There exists some dtypes in the result columns that is inferred.
        # Find them and map them to the default dtypes.
        specified_dtypes = {} if dtype is None else dtype
        unspecified_dtypes = {
            name: dtype
            for name, dtype in df._dtypes
            if name not in specified_dtypes
        }
        default_dtypes = {}

        for name, dt in unspecified_dtypes.items():
            if dt == np.dtype("i1"):
                # csv reader reads all null column as int8.
                # The dtype should remain int8.
                default_dtypes[name] = dt
            else:
                default_dtypes[name] = _maybe_convert_to_default_type(dt)
        df = df.astype(default_dtypes)

    return df


@_performance_tracking
@ioutils.doc_to_csv()
def to_csv(
    df,
    path_or_buf=None,
    sep=",",
    na_rep="",
    columns=None,
    header=True,
    index=True,
    encoding=None,
    compression=None,
    lineterminator="\n",
    chunksize=None,
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

    for col in df._data.columns:
        if isinstance(col, cudf.core.column.ListColumn):
            raise NotImplementedError(
                "Writing to csv format is not yet supported with "
                "list columns."
            )
        elif isinstance(col, cudf.core.column.StructColumn):
            raise NotImplementedError(
                "Writing to csv format is not yet supported with "
                "Struct columns."
            )

    # TODO: Need to typecast categorical columns to the underlying
    # categories dtype to write the actual data to csv. Remove this
    # workaround once following issue is fixed:
    # https://github.com/rapidsai/cudf/issues/6661
    if any(
        isinstance(col, cudf.core.column.CategoricalColumn)
        for col in df._data.columns
    ) or isinstance(df.index, cudf.CategoricalIndex):
        df = df.copy(deep=False)
        for col_name, col in df._data.items():
            if isinstance(col, cudf.core.column.CategoricalColumn):
                df._data[col_name] = col.astype(col.categories.dtype)

        if isinstance(df.index, cudf.CategoricalIndex):
            df.index = df.index.astype(df.index.categories.dtype)

    rows_per_chunk = chunksize if chunksize else len(df)

    if ioutils.is_fsspec_open_file(path_or_buf):
        with path_or_buf as file_obj:
            file_obj = ioutils.get_IOBase_writer(file_obj)
            libcudf.csv.write_csv(
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
        libcudf.csv.write_csv(
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
