# Copyright (c) 2023-2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture(params=[False, True, None])
def preserve_index(request):
    return request.param


def test_dataframe_to_arrow_preserve_index(preserve_index):
    df = cudf.DataFrame({"x": ["cat", "dog"] * 5})
    pf = df.to_pandas()
    expect = pa.Table.from_pandas(pf, preserve_index=preserve_index).schema
    got = df.to_arrow(preserve_index=preserve_index).schema
    assert expect == got


def test_datetime_to_arrow(datetime_types_as_str):
    data = pd.date_range("2000-01-01", "2000-01-02", freq="3600s")
    gdf = cudf.DataFrame({"timestamp": data.astype(datetime_types_as_str)})
    assert_eq(
        gdf, cudf.DataFrame.from_arrow(gdf.to_arrow(preserve_index=False))
    )


def test_arrow_pandas_compat(preserve_index):
    data = {"a": range(10), "b": range(10)}
    pdf = pd.DataFrame(data, index=pd.Index(np.arange(10), name="z"))
    gdf = cudf.DataFrame(data, index=cudf.Index(np.arange(10), name="z"))

    pdf_arrow_table = pa.Table.from_pandas(pdf, preserve_index=preserve_index)
    gdf_arrow_table = gdf.to_arrow(preserve_index=preserve_index)

    assert pa.Table.equals(pdf_arrow_table, gdf_arrow_table)

    gdf2 = cudf.DataFrame.from_arrow(pdf_arrow_table)
    pdf2 = pdf_arrow_table.to_pandas()

    assert_eq(pdf2, gdf2)
    pdf.columns.name = "abc"
    pdf_arrow_table = pa.Table.from_pandas(pdf, preserve_index=preserve_index)

    gdf2 = cudf.DataFrame.from_arrow(pdf_arrow_table)
    pdf2 = pdf_arrow_table.to_pandas()
    assert_eq(pdf2, gdf2)


@pytest.mark.parametrize(
    "index",
    [
        None,
        cudf.RangeIndex(3, name="a"),
        "a",
        "b",
        ["a", "b"],
        cudf.RangeIndex(0, 5, 2, name="a"),
    ],
)
def test_arrow_round_trip(preserve_index, index):
    data = {"a": [4, 5, 6], "b": ["cat", "dog", "bird"]}
    if isinstance(index, (list, str)):
        gdf = cudf.DataFrame(data).set_index(index)
    else:
        gdf = cudf.DataFrame(data, index=index)

    table = gdf.to_arrow(preserve_index=preserve_index)
    table_pd = pa.Table.from_pandas(
        gdf.to_pandas(), preserve_index=preserve_index
    )

    gdf_out = cudf.DataFrame.from_arrow(table)
    pdf_out = table_pd.to_pandas()

    assert_eq(gdf_out, pdf_out)
