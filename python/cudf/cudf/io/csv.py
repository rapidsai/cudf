# Copyright (c) 2018-20, NVIDIA CORPORATION.
from io import BytesIO, StringIO

from cudf import _lib as libcudf
from cudf._lib.nvtx import annotate
from cudf.utils import ioutils


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
    path=None,
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

    if path is None:
        raise ValueError("path/filename not provided")

    path_or_buf = ioutils.get_writer_filepath_or_buffer(
        path_or_data=path, mode="w", **kwargs
    )

    if index:
        from cudf import MultiIndex

        if not isinstance(df.index, MultiIndex):
            if df.index.name is None:
                df.index.name = ""
            if columns is not None:
                columns = columns.copy()
                columns.insert(0, df.index.name)
        df = df.reset_index()

    if columns is not None:
        try:
            df = df[columns]
        except KeyError:
            raise NameError(
                "Dataframe doesn't have the labels provided in columns"
            )

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
        )
