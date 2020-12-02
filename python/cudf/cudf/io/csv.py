# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from io import BytesIO, StringIO

from nvtx import annotate

import cudf
from cudf import _lib as libcudf
from cudf.utils import ioutils
from cudf.utils.dtypes import is_scalar


@annotate("READ_CSV", color="purple", domain="cudf_python")
@ioutils.doc_read_csv()
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
        path_or_data=filepath_or_buffer,
        compression=compression,
        iotypes=(BytesIO, StringIO),
        **kwargs,
    )

    if na_values is not None and is_scalar(na_values):
        na_values = [na_values]

    if keep_default_na is False:
        # TODO: Remove this error once the following issue is fixed:
        # https://github.com/rapidsai/cudf/issues/6680
        raise NotImplementedError(
            "keep_default_na=False is currently not supported, please refer "
            "to: https://github.com/rapidsai/cudf/issues/6680"
        )

    return libcudf.csv.read_csv(
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


@annotate("WRITE_CSV", color="purple", domain="cudf_python")
@ioutils.doc_to_csv()
def to_csv(
    df,
    path_or_buf=None,
    sep=",",
    na_rep="",
    columns=None,
    header=True,
    index=True,
    line_terminator="\n",
    chunksize=None,
    **kwargs,
):
    """{docstring}"""

    return_as_string = False
    if path_or_buf is None:
        path_or_buf = StringIO()
        return_as_string = True

    path_or_buf = ioutils.get_writer_filepath_or_buffer(
        path_or_data=path_or_buf, mode="w", **kwargs
    )

    if columns is not None:
        try:
            df = df[columns]
        except KeyError:
            raise NameError(
                "Dataframe doesn't have the labels provided in columns"
            )

    if sep == "-":
        # TODO: Remove this error once following issue is fixed:
        # https://github.com/rapidsai/cudf/issues/6699
        if any(
            isinstance(col, cudf.core.column.DatetimeColumn)
            for col in df._data.columns
        ):
            raise ValueError(
                "sep cannot be '-' when writing a datetime64 dtype to csv, "
                "refer to: https://github.com/rapidsai/cudf/issues/6699"
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
                df._data[col_name] = col.astype(col.cat().categories.dtype)

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
                line_terminator=line_terminator,
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
            line_terminator=line_terminator,
            rows_per_chunk=rows_per_chunk,
            index=index,
        )

    if return_as_string:
        path_or_buf.seek(0)
        return path_or_buf.read()
