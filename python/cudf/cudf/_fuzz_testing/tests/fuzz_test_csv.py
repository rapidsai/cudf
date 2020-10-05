# Copyright (c) 2020, NVIDIA CORPORATION.

import sys
from io import StringIO

import pandas as pd

import cudf
from cudf._fuzz_testing.csv import CSVReader, CSVWriter
from cudf._fuzz_testing.main import pythonfuzz
from cudf._fuzz_testing.utils import compare_content, run_test
from cudf.tests.utils import assert_eq


@pythonfuzz(data_handle=CSVReader)
def csv_reader_test(csv_buffer):
    pdf = pd.read_csv(StringIO(csv_buffer))
    gdf = cudf.read_csv(StringIO(csv_buffer))

    assert_eq(gdf, pdf)


@pythonfuzz(data_handle=CSVWriter)
def csv_writer_test(pdf):
    gdf = cudf.from_pandas(pdf)

    pd_buffer = pdf.to_csv()
    gd_buffer = gdf.to_csv()

    compare_content(pd_buffer, gd_buffer)
    actual = cudf.read_csv(StringIO(gd_buffer))
    expected = pd.read_csv(StringIO(pd_buffer))
    assert_eq(actual, expected)


@pythonfuzz(
    data_handle=CSVWriter,
    params={
        "sep": list([","]),
        "header": [True, False],
        "na_rep": [
            "",
            "<NA>",
            "NA",
            "_NA_",
            "__",
            "<<<<>>>>>",
            "--<>--",
            "-+><+-",
        ],
        "columns": None,
        "index": [True, False],
        "line_terminator": ["\n", "\r"],
        "chunksize": None,
    },
)
def csv_writer_test_params(
    pdf, sep, header, na_rep, columns, index, line_terminator, chunksize
):
    gdf = cudf.from_pandas(pdf)

    pd_buffer = pdf.to_csv(
        sep=sep,
        header=header,
        na_rep=na_rep,
        columns=columns,
        index=index,
        line_terminator=line_terminator,
        chunksize=chunksize,
    )
    gd_buffer = gdf.to_csv(
        sep=sep,
        header=header,
        na_rep=na_rep,
        columns=columns,
        index=index,
        line_terminator=line_terminator,
        chunksize=chunksize,
    )

    # compare_content(pd_buffer, gd_buffer)

    actual = cudf.read_csv(
        StringIO(gd_buffer),
        delimiter=sep,
        na_values=na_rep,
        lineterminator=line_terminator,
    )
    expected = pd.read_csv(
        StringIO(pd_buffer),
        delimiter=sep,
        na_values=na_rep,
        lineterminator=line_terminator,
    )
    if not header:
        # TODO: Remove renaming columns once the following bug is fixed:
        # https://github.com/rapidsai/cudf/issues/6418
        actual.columns = expected.columns

    assert_eq(actual, expected)


@pythonfuzz(
    data_handle=CSVReader,
    params={
        "dtype": None,
        "usecols": None,
        "header": None,
        "skiprows": None,
        "skipfooter": None,
        "nrows": None,
    },
)
def csv_reader_test_params(csv_buffer, dtype, header, skiprows):
    pdf = pd.read_csv(
        StringIO(csv_buffer), dtype=dtype, header=header, skiprows=skiprows
    )
    gdf = cudf.read_csv(
        StringIO(csv_buffer), dtype=dtype, header=header, skiprows=skiprows
    )

    assert_eq(gdf, pdf)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
