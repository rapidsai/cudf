# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import io
import sys

import pandas as pd

import cudf
from cudf._fuzz_testing.json import JSONReader, JSONWriter
from cudf._fuzz_testing.main import pythonfuzz
from cudf._fuzz_testing.utils import ALL_POSSIBLE_VALUES, run_test
from cudf.testing import assert_eq


@pythonfuzz(data_handle=JSONReader)
def json_reader_test(json_buffer):
    pdf = pd.read_json(io.StringIO(json_buffer), orient="records", lines=True)
    # Difference in behaviour with pandas
    # cudf reads column as strings only.
    pdf.columns = pdf.columns.astype("str")
    gdf = cudf.read_json(io.StringIO(json_buffer), engine="cudf", lines=True)

    assert_eq(gdf, pdf)


@pythonfuzz(data_handle=JSONReader, params={"dtype": ALL_POSSIBLE_VALUES})
def json_reader_test_params(json_buffer, dtype):
    pdf = pd.read_json(json_buffer, dtype=dtype, orient="records", lines=True)
    pdf.columns = pdf.columns.astype("str")

    gdf = cudf.read_json(json_buffer, dtype=dtype, engine="cudf", lines=True)

    assert_eq(gdf, pdf)


@pythonfuzz(data_handle=JSONWriter)
def json_writer_test(pdf):
    gdf = cudf.from_pandas(pdf)

    pdf_buffer = pdf.to_json(lines=True, orient="records")
    gdf_buffer = gdf.to_json(lines=True, orient="records")

    # TODO: Uncomment once this is fixed:
    # https://github.com/rapidsai/cudf/issues/6429
    # compare_content(pdf_buffer, gdf_buffer)

    actual = cudf.read_json(
        gdf_buffer, engine="cudf", lines=True, orient="records"
    )
    expected = pd.read_json(pdf_buffer, lines=True, orient="records")
    expected.columns = expected.columns.astype("str")
    assert_eq(actual, expected)


@pythonfuzz(
    data_handle=JSONWriter,
    params={
        "compression": ["gzip", "bz2", "zip", "xz", None],
        "dtype": ALL_POSSIBLE_VALUES,
    },
)
def json_writer_test_params(pdf, compression, dtype):
    gdf = cudf.from_pandas(pdf)

    pdf_buffer = pdf.to_json(
        lines=True, orient="records", compression=compression
    )
    gdf_buffer = gdf.to_json(
        lines=True, orient="records", compression=compression
    )

    # TODO: Uncomment once this is fixed:
    # https://github.com/rapidsai/cudf/issues/6429
    # compare_content(pdf_buffer, gdf_buffer)

    actual = cudf.read_json(
        io.StringIO(gdf_buffer),
        engine="cudf",
        lines=True,
        orient="records",
        dtype=dtype,
    )
    expected = pd.read_json(
        io.StringIO(pdf_buffer), lines=True, orient="records", dtype=dtype
    )

    # Difference in behaviour with pandas
    # cudf reads column as strings only.
    expected.columns = expected.columns.astype("str")
    assert_eq(actual, expected)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
