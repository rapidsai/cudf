# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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


def test_dataframe_list_round_trip():
    data = [{"text": "hello", "list_col": np.asarray([1, 2], dtype="uint32")}]
    cudf_arrow = cudf.DataFrame(data).to_arrow()
    pdf_arrow = pa.Table.from_pandas(pd.DataFrame(data))

    for metadata in (
        None,
        pdf_arrow.schema.metadata,
        cudf_arrow.schema.metadata,
    ):
        schema = pa.schema(
            [
                pa.field("text", pa.string()),
                pa.field("list_col", pa.list_(pa.uint32())),
            ],
            metadata=metadata,
        )

        data = {"text": ["asd", "pqr"], "list_col": [[1, 2, 3], [4, 5]]}

        table = pa.Table.from_pydict(data, schema=schema)
        assert_eq(table.to_pandas(), pd.DataFrame(data))


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


@pytest.mark.parametrize("nelem", [0, 2])
def test_to_arrow(nelem, all_supported_types_as_str):
    if all_supported_types_as_str in {"category", "str"}:
        pytest.skip(f"Test not applicable with {all_supported_types_as_str}")
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "a": rng.integers(0, 1000, nelem).astype(
                all_supported_types_as_str
            ),
            "b": rng.integers(0, 1000, nelem).astype(
                all_supported_types_as_str
            ),
        }
    )
    gdf = cudf.DataFrame(df)

    pa_df = pa.Table.from_pandas(
        df, preserve_index=False
    ).replace_schema_metadata(None)

    pa_gdf = gdf.to_arrow(preserve_index=False).replace_schema_metadata(None)

    assert isinstance(pa_gdf, pa.Table)
    assert pa.Table.equals(pa_df, pa_gdf)

    pa_s = pa.Array.from_pandas(df.a)
    pa_gs = gdf["a"].to_arrow()

    assert isinstance(pa_gs, pa.Array)
    assert pa.Array.equals(pa_s, pa_gs)

    pa_i = pa.Array.from_pandas(df.index)
    pa_gi = gdf.index.to_arrow()

    assert isinstance(pa_gi, pa.Array)
    assert pa.Array.equals(pa_i, pa_gi)


def test_to_arrow_categorical():
    df = pd.DataFrame({"a": pd.Series(["a", "b", "c"], dtype="category")})
    gdf = cudf.DataFrame(df)

    pa_df = pa.Table.from_pandas(
        df, preserve_index=False
    ).replace_schema_metadata(None)
    pa_gdf = gdf.to_arrow(preserve_index=False).replace_schema_metadata(None)

    assert isinstance(pa_gdf, pa.Table)
    assert pa.Table.equals(pa_df, pa_gdf)

    pa_s = pa.Array.from_pandas(df.a)
    pa_gs = gdf["a"].to_arrow()

    assert isinstance(pa_gs, pa.Array)
    assert pa.Array.equals(pa_s, pa_gs)


@pytest.mark.parametrize(
    "data",
    [
        {0: [1, 2, 3], 2: [10, 11, 23]},
        {("a", "b"): [1, 2, 3], ("2",): [10, 11, 23]},
    ],
)
def test_non_string_column_name_to_arrow(data):
    df = cudf.DataFrame(data)

    expected = df.to_arrow()
    actual = pa.Table.from_pandas(df.to_pandas())

    assert expected.equals(actual)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [{"one": 3, "two": 4, "three": 10}]},
        {
            "left-a": [0, 1, 2],
            "a": [{"x": 0.23, "y": 43}, None, {"x": 23.9, "y": 4.3}],
            "right-a": ["abc", "def", "ghi"],
        },
        {
            "left-a": [{"a": 1}, None, None],
            "a": [
                {"one": 324, "two": 23432, "three": 324},
                None,
                {"one": 3.24, "two": 1, "three": 324},
            ],
            "right-a": ["abc", "def", "ghi"],
        },
    ],
)
def test_dataframe_roundtrip_arrow_struct_dtype(data):
    gdf = cudf.DataFrame(data)
    table = gdf.to_arrow()
    expected = cudf.DataFrame.from_arrow(table)

    assert_eq(gdf, expected)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[1], [2], [3]]},
        {
            "left-a": [0, 1, 2],
            "a": [[1], None, [3]],
            "right-a": ["abc", "def", "ghi"],
        },
        {
            "left-a": [[], None, None],
            "a": [[1], None, [3]],
            "right-a": ["abc", "def", "ghi"],
        },
    ],
)
def test_dataframe_roundtrip_arrow_list_dtype(data):
    gdf = cudf.DataFrame(data)
    table = gdf.to_arrow()
    expected = cudf.DataFrame.from_arrow(table)

    assert_eq(gdf, expected)
