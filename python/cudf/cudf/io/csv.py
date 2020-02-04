# Copyright (c) 2018, NVIDIA CORPORATION.

from io import BytesIO, IOBase, StringIO

import cudf._lib as libcudf
from cudf.utils import ioutils


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
        filepath_or_buffer, compression, (BytesIO, StringIO), **kwargs
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


@ioutils.doc_to_csv()
def to_csv(
    df,
    path=None,
    sep=",",
    na_rep="",
    columns=None,
    header=True,
    index=True,
    line_terminator="\n",
    chunksize=None,
):
    """{docstring}"""
    if index:
        from cudf import MultiIndex

        if not isinstance(df.index, MultiIndex):
            if df.index.name is None:
                df.index.name = ""
            if columns is not None:
                columns = columns.copy()
                columns.insert(0, df.index.name)
        df = df.reset_index()
    rows_per_chunk = chunksize if chunksize else len(df)

    if isinstance(path, IOBase):
        path = path.name

    return libcudf.csv.write_csv(
        cols=df._data,
        path=path,
        sep=sep,
        na_rep=na_rep,
        columns=columns,
        header=header,
        line_terminator=line_terminator,
        rows_per_chunk=rows_per_chunk,
    )
