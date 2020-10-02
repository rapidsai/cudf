# Copyright (c) 2020, NVIDIA CORPORATION.

import sys

import pandas as pd

import cudf
from cudf.testing.json import JSONReader, JSONWriter
from cudf.testing.main import pythonfuzz
from cudf.testing.utils import compare_content, run_test
from cudf.tests.utils import assert_eq


@pythonfuzz(data_handle=JSONReader)
def json_reader_test(json_buffer):
    pdf = pd.read_json(json_buffer)
    gdf = cudf.read_json(json_buffer)

    assert_eq(gdf, pdf)


@pythonfuzz(data_handle=JSONWriter)
def json_writer_test(gdf):
    pdf = gdf.to_pandas()

    pdf_buffer = pdf.to_json(lines=True, orient="records")
    gdf_buffer = gdf.to_json(lines=True, orient="records")

    compare_content(pdf_buffer, gdf_buffer)

    actual = cudf.read_json(gdf_buffer, lines=True, orient="records")
    expected = pd.read_json(pdf_buffer, lines=True, orient="records")
    assert_eq(actual, expected)


if __name__ == "__main__":
    run_test(globals(), sys.argv)
