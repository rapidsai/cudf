# Copyright (c) 2025, NVIDIA CORPORATION.

from contextlib import nullcontext as does_not_raise

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


def test_init_via_list_of_tuples():
    data = [
        (5, "cats", "jump", np.nan),
        (2, "dogs", "dig", 7.5),
        (3, "cows", "moo", -2.1, "occasionally"),
    ]

    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("columns", [["a", "b"], pd.Series(["a", "b"])])
def test_init_via_list_of_series(columns):
    data = [pd.Series([1, 2]), pd.Series([3, 4])]

    pdf = cudf.DataFrame(data, columns=columns)
    gdf = cudf.DataFrame(data, columns=columns)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("index", [None, [0, 1, 2]])
def test_init_with_missing_columns(index):
    """Test initialization when columns and data keys are disjoint."""
    data = {"a": [1, 2, 3], "b": [2, 3, 4]}
    columns = ["c", "d"]

    pdf = cudf.DataFrame(data, columns=columns, index=index)
    gdf = cudf.DataFrame(data, columns=columns, index=index)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("rows", [0, 1, 2, 100])
def test_init_via_list_of_empty_tuples(rows):
    data = [()] * rows

    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "dict_of_series",
    [
        {"a": pd.Series([1.0, 2.0, 3.0])},
        {"a": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6])},
        {
            "a": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6]),
            "b": pd.Series([1.0, 2.0, 4.0], index=[1, 2, 3]),
        },
        {"a": [1, 2, 3], "b": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6])},
        {
            "a": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
            "b": pd.Series([1.0, 2.0, 4.0], index=["c", "d", "e"]),
        },
        {
            "a": pd.Series(
                ["a", "b", "c"],
                index=pd.MultiIndex.from_tuples([(1, 2), (1, 3), (2, 3)]),
            ),
            "b": pd.Series(
                ["a", " b", "d"],
                index=pd.MultiIndex.from_tuples([(1, 2), (1, 3), (2, 3)]),
            ),
        },
    ],
)
def test_init_from_series_align(dict_of_series):
    pdf = pd.DataFrame(dict_of_series)
    gdf = cudf.DataFrame(dict_of_series)

    assert_eq(pdf, gdf)

    for key in dict_of_series:
        if isinstance(dict_of_series[key], pd.Series):
            dict_of_series[key] = cudf.Series(dict_of_series[key])

    gdf = cudf.DataFrame(dict_of_series)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    ("dict_of_series", "expectation"),
    [
        (
            {
                "a": pd.Series(["a", "b", "c"], index=[4, 4, 5]),
                "b": pd.Series(["a", "b", "c"], index=[4, 5, 6]),
            },
            pytest.raises(
                ValueError, match="Cannot align indices with non-unique values"
            ),
        ),
        (
            {
                "a": pd.Series(["a", "b", "c"], index=[4, 4, 5]),
                "b": pd.Series(["a", "b", "c"], index=[4, 4, 5]),
            },
            does_not_raise(),
        ),
    ],
)
def test_init_from_series_align_nonunique(dict_of_series, expectation):
    with expectation:
        gdf = cudf.DataFrame(dict_of_series)

    if expectation == does_not_raise():
        pdf = pd.DataFrame(dict_of_series)
        assert_eq(pdf, gdf)


def test_init_unaligned_with_index():
    pdf = pd.DataFrame(
        {
            "a": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6]),
            "b": pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3]),
        },
        index=[7, 8, 9],
    )
    gdf = cudf.DataFrame(
        {
            "a": cudf.Series([1.0, 2.0, 3.0], index=[4, 5, 6]),
            "b": cudf.Series([1.0, 2.0, 3.0], index=[1, 2, 3]),
        },
        index=[7, 8, 9],
    )

    assert_eq(pdf, gdf, check_dtype=False)


def test_init_series_list_columns_unsort():
    pseries = [
        pd.Series(i, index=["b", "a", "c"], name=str(i)) for i in range(3)
    ]
    gseries = [
        cudf.Series(i, index=["b", "a", "c"], name=str(i)) for i in range(3)
    ]
    pdf = pd.DataFrame(pseries)
    gdf = cudf.DataFrame(gseries)
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("nelem", [0, 10])
@pytest.mark.parametrize("nchunks", [1, 5])
def test_from_arrow_chunked_arrays(nelem, nchunks, numeric_types_as_str):
    rng = np.random.default_rng(seed=0)
    np_list_data = [
        rng.integers(0, 100, nelem).astype(numeric_types_as_str)
        for i in range(nchunks)
    ]
    pa_chunk_array = pa.chunked_array(np_list_data)

    expect = pa_chunk_array.to_pandas()
    got = cudf.Series(pa_chunk_array)

    assert_eq(expect, got)

    np_list_data2 = [
        rng.integers(0, 100, nelem).astype(numeric_types_as_str)
        for i in range(nchunks)
    ]
    pa_chunk_array2 = pa.chunked_array(np_list_data2)
    pa_table = pa.Table.from_arrays(
        [pa_chunk_array, pa_chunk_array2], names=["a", "b"]
    )

    expect = pa_table.to_pandas()
    got = cudf.DataFrame.from_arrow(pa_table)

    assert_eq(expect, got)


def test_1row_arrow_table():
    data = [pa.array([0]), pa.array([1])]
    batch = pa.RecordBatch.from_arrays(data, ["f0", "f1"])
    table = pa.Table.from_batches([batch])

    expect = table.to_pandas()
    got = cudf.DataFrame.from_arrow(table)
    assert_eq(expect, got)


def test_arrow_handle_no_index_name():
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    gdf_arrow = gdf.to_arrow()
    pdf_arrow = pa.Table.from_pandas(pdf)
    assert pa.Table.equals(pdf_arrow, gdf_arrow)

    got = cudf.DataFrame.from_arrow(gdf_arrow)
    expect = pdf_arrow.to_pandas()
    assert_eq(expect, got)


def test_pandas_non_contiguious():
    rng = np.random.default_rng(seed=0)
    arr1 = rng.random(size=(5000, 10))
    assert arr1.flags["C_CONTIGUOUS"] is True
    df = pd.DataFrame(arr1)
    for col in df.columns:
        assert df[col].values.flags["C_CONTIGUOUS"] is False

    gdf = cudf.DataFrame.from_pandas(df)
    assert_eq(gdf.to_pandas(), df)


def test_from_records(numeric_types_as_str):
    h_ary = np.ndarray(shape=(10, 4), dtype=numeric_types_as_str)
    rec_ary = h_ary.view(np.recarray)

    gdf = cudf.DataFrame.from_records(rec_ary, columns=["a", "b", "c", "d"])
    df = pd.DataFrame.from_records(rec_ary, columns=["a", "b", "c", "d"])
    assert isinstance(gdf, cudf.DataFrame)
    assert_eq(df, gdf)

    gdf = cudf.DataFrame.from_records(rec_ary)
    df = pd.DataFrame.from_records(rec_ary)
    assert isinstance(gdf, cudf.DataFrame)
    assert_eq(df, gdf)


@pytest.mark.parametrize("columns", [None, ["first", "second", "third"]])
@pytest.mark.parametrize(
    "index",
    [
        None,
        ["first", "second"],
        "name",
        "age",
        "weight",
        [10, 11],
        ["abc", "xyz"],
    ],
)
def test_from_records_index(columns, index):
    rec_ary = np.array(
        [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
        dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
    )
    gdf = cudf.DataFrame.from_records(rec_ary, columns=columns, index=index)
    df = pd.DataFrame.from_records(rec_ary, columns=columns, index=index)
    assert isinstance(gdf, cudf.DataFrame)
    assert_eq(df, gdf)


def test_dataframe_construction_from_cupy_arrays():
    h_ary = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    d_ary = cp.asarray(h_ary)

    gdf = cudf.DataFrame(d_ary, columns=["a", "b", "c"])
    df = pd.DataFrame(h_ary, columns=["a", "b", "c"])
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary)
    df = pd.DataFrame(h_ary)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary, index=["a", "b"])
    df = pd.DataFrame(h_ary, index=["a", "b"])
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary)
    gdf = gdf.set_index(keys=0, drop=False)
    df = pd.DataFrame(h_ary)
    df = df.set_index(keys=0, drop=False)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary)
    gdf = gdf.set_index(keys=1, drop=False)
    df = pd.DataFrame(h_ary)
    df = df.set_index(keys=1, drop=False)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)


def test_dataframe_cupy_wrong_dimensions():
    d_ary = cp.empty((2, 3, 4), dtype=np.int32)
    with pytest.raises(
        ValueError, match="records dimension expected 1 or 2 but found: 3"
    ):
        cudf.DataFrame(d_ary)


def test_dataframe_cupy_array_wrong_index():
    d_ary = cp.empty((2, 3), dtype=np.int32)

    with pytest.raises(ValueError):
        cudf.DataFrame(d_ary, index=["a"])

    with pytest.raises(TypeError):
        cudf.DataFrame(d_ary, index="a")


def test_index_in_dataframe_constructor():
    a = pd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])
    b = cudf.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])

    assert_eq(a, b)
    assert_eq(a.loc[4:], b.loc[4:])


@pytest.mark.parametrize("nelem", [0, 2])
def test_from_arrow(nelem, all_supported_types_as_str):
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
    padf = pa.Table.from_pandas(
        df, preserve_index=False
    ).replace_schema_metadata(None)
    gdf = cudf.DataFrame.from_arrow(padf)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    s = pa.Array.from_pandas(df.a)
    gs = cudf.Series.from_arrow(s)
    assert isinstance(gs, cudf.Series)

    # For some reason PyArrow to_pandas() converts to numpy array and has
    # better type compatibility
    np.testing.assert_array_equal(s.to_pandas(), gs.to_numpy())


def test_from_arrow_chunked_categories():
    # Verify that categories are properly deduplicated across chunked arrays.
    indices = pa.array([0, 1, 0, 1, 2, 0, None, 2])
    dictionary = pa.array(["foo", "bar", "baz"])
    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    chunked_array = pa.chunked_array([dict_array, dict_array])
    table = pa.table({"a": chunked_array})
    df = cudf.DataFrame.from_arrow(table)
    final_dictionary = df["a"].dtype.categories.to_arrow().to_pylist()
    assert sorted(final_dictionary) == sorted(dictionary.to_pylist())


def test_from_scalar_typing(request, all_supported_types_as_str):
    if all_supported_types_as_str in {"category", "str"}:
        pytest.skip(f"Test not applicable with {all_supported_types_as_str}")
    request.applymarker(
        pytest.mark.xfail(
            all_supported_types_as_str
            in {"timedelta64[ms]", "timedelta64[us]", "timedelta64[ns]"},
            reason=f"{all_supported_types_as_str} incorrectly results in timedelta64[s]",
        )
    )
    rng = np.random.default_rng(seed=0)
    if all_supported_types_as_str == "datetime64[ms]":
        scalar = (
            np.dtype("int64").type(rng.integers(0, 5)).astype("datetime64[ms]")
        )
    elif all_supported_types_as_str.startswith("datetime64"):
        scalar = np.datetime64("2020-01-01").astype("datetime64[ms]")
        all_supported_types_as_str = "datetime64[ms]"
    else:
        scalar = np.dtype(all_supported_types_as_str).type(rng.integers(0, 5))

    gdf = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": scalar,
        }
    )
    assert gdf["b"].dtype == np.dtype(all_supported_types_as_str)
    assert len(gdf["b"]) == len(gdf["a"])
