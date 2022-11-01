# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import io
import sys

import cudf
from cudf._fuzz_testing.main import pythonfuzz
from cudf._fuzz_testing.orc import OrcReader, OrcWriter
from cudf._fuzz_testing.utils import (
    ALL_POSSIBLE_VALUES,
    compare_dataframe,
    orc_to_pandas,
    run_test,
)


@pythonfuzz(
    data_handle=OrcReader,
    params={
        "columns": ALL_POSSIBLE_VALUES,
        "skiprows": ALL_POSSIBLE_VALUES,
        "num_rows": ALL_POSSIBLE_VALUES,
        "use_index": ALL_POSSIBLE_VALUES,
    },
)
def orc_reader_test(input_tuple, columns, skiprows, num_rows, use_index):
    pdf, file_buffer = input_tuple
    expected_pdf = pdf.iloc[skiprows:]
    if num_rows is not None:
        expected_pdf = expected_pdf.head(num_rows)
    if skiprows is not None or num_rows is not None:
        expected_pdf.reset_index(drop=True, inplace=True)
    if columns is not None and len(columns) > 0:
        # ORC reader picks columns if only
        # there are any elements in `columns`
        expected_pdf = expected_pdf[columns]
    if use_index is False:
        expected_pdf.reset_index(drop=True, inplace=True)

    gdf = cudf.read_orc(
        io.BytesIO(file_buffer),
        columns=columns,
        skiprows=skiprows,
        num_rows=num_rows,
        use_index=use_index,
    )

    compare_dataframe(expected_pdf, gdf)


@pythonfuzz(
    data_handle=OrcReader,
    params={"columns": ALL_POSSIBLE_VALUES, "stripes": ALL_POSSIBLE_VALUES},
)
def orc_reader_stripes_test(input_tuple, columns, stripes):
    _, file_buffer = input_tuple
    expected_pdf = orc_to_pandas(
        file_io_obj=io.BytesIO(file_buffer), stripes=stripes
    )

    if columns is not None and len(columns) > 0:
        # ORC reader picks columns if only
        # there are any elements in `columns`
        expected_pdf = expected_pdf[columns]

    gdf = cudf.read_orc(
        io.BytesIO(file_buffer), columns=columns, stripes=stripes
    )

    compare_dataframe(expected_pdf, gdf)


@pythonfuzz(
    data_handle=OrcWriter,
    params={
        "compression": [None, "snappy"],
        "enable_statistics": ["NONE", "STRIPE", "ROWGROUP"],
    },
)
def orc_writer_test(pdf, compression, enable_statistics):
    file_to_strore = io.BytesIO()

    gdf = cudf.from_pandas(pdf)

    gdf.to_orc(
        file_to_strore,
        compression=compression,
        enable_statistics=enable_statistics,
    )
    file_to_strore.seek(0)

    actual_df = cudf.read_orc(file_to_strore)

    compare_dataframe(pdf, actual_df)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
