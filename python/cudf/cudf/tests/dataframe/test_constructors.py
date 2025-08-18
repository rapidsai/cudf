# Copyright (c) 2025, NVIDIA CORPORATION.

from contextlib import nullcontext as does_not_raise

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numba import cuda

import cudf
from cudf.core.column.column import as_column
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


def test_dataframe_construction_from_cp_arrays():
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


def test_dataframe_cp_wrong_dimensions():
    d_ary = cp.empty((2, 3, 4), dtype=np.int32)
    with pytest.raises(
        ValueError, match="records dimension expected 1 or 2 but found: 3"
    ):
        cudf.DataFrame(d_ary)


def test_dataframe_cp_array_wrong_index():
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


@pytest.mark.parametrize(
    "data",
    [
        {"a": [np.nan, 1, 2], "b": [None, None, None]},
        {"a": [1, 2, np.nan, 2], "b": [np.nan, np.nan, np.nan, np.nan]},
        {
            "a": [1, 2, np.nan, 2, None],
            "b": [np.nan, np.nan, None, np.nan, np.nan],
        },
        {"a": [1, 2, 2, None, 1.1], "b": [1, 2.2, 3, None, 5]},
    ],
)
def test_dataframe_constructor_nan_as_null(data, nan_as_null):
    actual = cudf.DataFrame(data, nan_as_null=nan_as_null)

    if nan_as_null:
        assert (
            not (
                actual.astype("float").replace(
                    cudf.Series([np.nan], nan_as_null=False), cudf.Series([-1])
                )
                == -1
            )
            .any()
            .any()
        )
    else:
        actual = actual.select_dtypes(exclude=["object"])
        assert (actual.replace(np.nan, -1) == -1).any().any()


@pytest.mark.parametrize(
    "data,columns,index",
    [
        (pd.Series([1, 2, 3]), None, None),
        (pd.Series(["a", "b", None, "c"], name="abc"), None, None),
        (
            pd.Series(["a", "b", None, "c"], name="abc"),
            ["abc", "b"],
            [1, 2, 3],
        ),
    ],
)
def test_dataframe_init_from_series(data, columns, index):
    expected = pd.DataFrame(data, columns=columns, index=index)
    actual = cudf.DataFrame(data, columns=columns, index=index)

    assert_eq(
        expected,
        actual,
        check_index_type=len(expected) != 0,
    )


@pytest.mark.parametrize(
    "dtype,expected_upcast_type,error",
    [
        (
            "float32",
            np.dtype("float32"),
            None,
        ),
        (
            "float16",
            None,
            TypeError,
        ),
        (
            "float64",
            np.dtype("float64"),
            None,
        ),
        (
            "float128",
            None,
            ValueError,
        ),
    ],
)
def test_from_pandas_unsupported_types(dtype, expected_upcast_type, error):
    data = pd.Series([1.1, 0.55, -1.23], dtype=dtype)
    pdf = pd.DataFrame({"one_col": data})
    if error is not None:
        with pytest.raises(error):
            cudf.from_pandas(data)

        with pytest.raises(error):
            cudf.Series(data)

        with pytest.raises(error):
            cudf.from_pandas(pdf)

        with pytest.raises(error):
            cudf.DataFrame(pdf)
    else:
        df = cudf.from_pandas(data)

        assert_eq(data, df, check_dtype=False)
        assert df.dtype == expected_upcast_type

        df = cudf.Series(data)
        assert_eq(data, df, check_dtype=False)
        assert df.dtype == expected_upcast_type

        df = cudf.from_pandas(pdf)
        assert_eq(pdf, df, check_dtype=False)
        assert df["one_col"].dtype == expected_upcast_type

        df = cudf.DataFrame(pdf)
        assert_eq(pdf, df, check_dtype=False)
        assert df["one_col"].dtype == expected_upcast_type


@pytest.mark.parametrize("index", [None, "a", ["a", "b"]])
def test_from_pandas_nan_as_null(nan_as_null, index):
    data = [np.nan, 2.0, 3.0]

    if index is None:
        pdf = pd.DataFrame({"a": data, "b": data})
        expected = cudf.DataFrame(
            {
                "a": as_column(data, nan_as_null=nan_as_null),
                "b": as_column(data, nan_as_null=nan_as_null),
            }
        )
    else:
        pdf = pd.DataFrame({"a": data, "b": data}).set_index(index)
        expected = cudf.DataFrame(
            {
                "a": as_column(data, nan_as_null=nan_as_null),
                "b": as_column(data, nan_as_null=nan_as_null),
            }
        )
        expected = cudf.DataFrame(
            {
                "a": as_column(data, nan_as_null=nan_as_null),
                "b": as_column(data, nan_as_null=nan_as_null),
            }
        )
        expected = expected.set_index(index)

    got = cudf.from_pandas(pdf, nan_as_null=nan_as_null)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data,columns",
    [
        ([1, 2, 3, 100, 112, 35464], ["a"]),
        (range(100), None),
        (
            [],
            None,
        ),
        ((-10, 21, 32, 32, 1, 2, 3), ["p"]),
        (
            (),
            None,
        ),
        ([[1, 2, 3], [1, 2, 3]], ["col1", "col2", "col3"]),
        ([range(100), range(100)], ["range" + str(i) for i in range(100)]),
        (((1, 2, 3), (1, 2, 3)), ["tuple0", "tuple1", "tuple2"]),
        ([[1, 2, 3]], ["list col1", "list col2", "list col3"]),
        ([[1, 2, 3]], pd.Index(["col1", "col2", "col3"], name="rapids")),
        ([range(100)], ["range" + str(i) for i in range(100)]),
        (((1, 2, 3),), ["k1", "k2", "k3"]),
    ],
)
def test_dataframe_init_1d_list(data, columns):
    expect = pd.DataFrame(data, columns=columns)
    actual = cudf.DataFrame(data, columns=columns)

    assert_eq(
        expect,
        actual,
        check_index_type=len(data) != 0,
    )

    expect = pd.DataFrame(data, columns=None)
    actual = cudf.DataFrame(data, columns=None)

    assert_eq(
        expect,
        actual,
        check_index_type=len(data) != 0,
    )


@pytest.mark.parametrize("dtype", ["int64", "str"])
def test_dataframe_from_dictionary_series_same_name_index(dtype):
    pd_idx1 = pd.Index([1, 2, 0], name="test_index").astype(dtype)
    pd_idx2 = pd.Index([2, 0, 1], name="test_index").astype(dtype)
    pd_series1 = pd.Series([1, 2, 3], index=pd_idx1)
    pd_series2 = pd.Series([1, 2, 3], index=pd_idx2)

    gd_idx1 = cudf.from_pandas(pd_idx1)
    gd_idx2 = cudf.from_pandas(pd_idx2)
    gd_series1 = cudf.Series([1, 2, 3], index=gd_idx1)
    gd_series2 = cudf.Series([1, 2, 3], index=gd_idx2)

    expect = pd.DataFrame({"a": pd_series1, "b": pd_series2})
    got = cudf.DataFrame({"a": gd_series1, "b": gd_series2})

    if dtype == "str":
        # Pandas actually loses its index name erroneously here...
        expect.index.name = "test_index"

    assert_eq(expect, got)
    assert expect.index.names == got.index.names


def test_init_multiindex_from_dict():
    pdf = pd.DataFrame({("a", "b"): [1]})
    gdf = cudf.DataFrame({("a", "b"): [1]})
    assert_eq(pdf, gdf)
    assert_eq(pdf.columns, gdf.columns)


def test_change_column_dtype_in_empty():
    pdf = pd.DataFrame({"a": [], "b": []})
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf, gdf)
    pdf["b"] = pdf["b"].astype("int64")
    gdf["b"] = gdf["b"].astype("int64")
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data,cols,index",
    [
        (
            np.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "b"],
            ["a", "b", "c", "d"],
        ),
        (
            np.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "b"],
            [0, 20, 30, 10],
        ),
        (
            np.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "b"],
            [0, 1, 2, 3],
        ),
        (np.array([11, 123, -2342, 232]), ["a"], [1, 2, 11, 12]),
        (np.array([11, 123, -2342, 232]), ["a"], ["khsdjk", "a", "z", "kk"]),
        (
            cp.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "z"],
            ["a", "z", "a", "z"],
        ),
        (cp.array([11, 123, -2342, 232]), ["z"], [0, 1, 1, 0]),
        (cp.array([11, 123, -2342, 232]), ["z"], [1, 2, 3, 4]),
        (cp.array([11, 123, -2342, 232]), ["z"], ["a", "z", "d", "e"]),
        (
            np.random.default_rng(seed=0).standard_normal(size=(2, 4)),
            ["a", "b", "c", "d"],
            ["a", "b"],
        ),
        (
            np.random.default_rng(seed=0).standard_normal(size=(2, 4)),
            ["a", "b", "c", "d"],
            [1, 0],
        ),
        (cp.random.randn(2, 4), ["a", "b", "c", "d"], ["a", "b"]),
        (cp.random.randn(2, 4), ["a", "b", "c", "d"], [1, 0]),
    ],
)
def test_dataframe_init_from_arrays_cols(data, cols, index):
    gd_data = data
    if isinstance(data, cp.ndarray):
        # pandas can't handle cp arrays in general
        pd_data = data.get()

        # additional test for building DataFrame with gpu array whose
        # cuda array interface has no `descr` attribute
        numba_data = cuda.as_cuda_array(data)
    else:
        pd_data = data
        numba_data = None

    # verify with columns & index
    pdf = pd.DataFrame(pd_data, columns=cols, index=index)
    gdf = cudf.DataFrame(gd_data, columns=cols, index=index)

    assert_eq(pdf, gdf, check_dtype=False)

    # verify with columns
    pdf = pd.DataFrame(pd_data, columns=cols)
    gdf = cudf.DataFrame(gd_data, columns=cols)

    assert_eq(pdf, gdf, check_dtype=False)

    pdf = pd.DataFrame(pd_data)
    gdf = cudf.DataFrame(gd_data)

    assert_eq(pdf, gdf, check_dtype=False)

    if numba_data is not None:
        gdf = cudf.DataFrame(numba_data)
        assert_eq(pdf, gdf, check_dtype=False)


@pytest.mark.parametrize(
    "data, orient, dtype, columns",
    [
        (
            {"col_1": [3, 2, 1, 0], "col_2": [3, 2, 1, 0]},
            "columns",
            None,
            None,
        ),
        ({"col_1": [3, 2, 1, 0], "col_2": [3, 2, 1, 0]}, "index", None, None),
        (
            {"col_1": [None, 2, 1, 0], "col_2": [3, None, 1, 0]},
            "index",
            None,
            ["A", "B", "C", "D"],
        ),
        (
            {
                "col_1": ["ab", "cd", "ef", "gh"],
                "col_2": ["zx", "one", "two", "three"],
            },
            "index",
            None,
            ["A", "B", "C", "D"],
        ),
        (
            {
                "index": [("a", "b"), ("a", "c")],
                "columns": [("x", 1), ("y", 2)],
                "data": [[1, 3], [2, 4]],
                "index_names": ["n1", "n2"],
                "column_names": ["z1", "z2"],
            },
            "tight",
            "float64",
            None,
        ),
    ],
)
def test_dataframe_from_dict(data, orient, dtype, columns):
    expected = pd.DataFrame.from_dict(
        data=data, orient=orient, dtype=dtype, columns=columns
    )

    actual = cudf.DataFrame.from_dict(
        data=data, orient=orient, dtype=dtype, columns=columns
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize("dtype", ["int64", "str", None])
def test_dataframe_from_dict_transposed(dtype):
    pd_data = {"a": [3, 2, 1, 0], "col_2": [3, 2, 1, 0]}
    gd_data = {key: cudf.Series(val) for key, val in pd_data.items()}

    expected = pd.DataFrame.from_dict(pd_data, orient="index", dtype=dtype)
    actual = cudf.DataFrame.from_dict(gd_data, orient="index", dtype=dtype)

    gd_data = {key: cp.asarray(val) for key, val in pd_data.items()}
    actual = cudf.DataFrame.from_dict(gd_data, orient="index", dtype=dtype)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "pd_data, gd_data, orient, dtype, columns",
    [
        (
            {"col_1": np.array([3, 2, 1, 0]), "col_2": np.array([3, 2, 1, 0])},
            {
                "col_1": cp.array([3, 2, 1, 0]),
                "col_2": cp.array([3, 2, 1, 0]),
            },
            "columns",
            None,
            None,
        ),
        (
            {"col_1": np.array([3, 2, 1, 0]), "col_2": np.array([3, 2, 1, 0])},
            {
                "col_1": cp.array([3, 2, 1, 0]),
                "col_2": cp.array([3, 2, 1, 0]),
            },
            "index",
            None,
            None,
        ),
        (
            {
                "col_1": np.array([None, 2, 1, 0]),
                "col_2": np.array([3, None, 1, 0]),
            },
            {
                "col_1": cp.array([np.nan, 2, 1, 0]),
                "col_2": cp.array([3, np.nan, 1, 0]),
            },
            "index",
            None,
            ["A", "B", "C", "D"],
        ),
        (
            {
                "col_1": np.array(["ab", "cd", "ef", "gh"]),
                "col_2": np.array(["zx", "one", "two", "three"]),
            },
            {
                "col_1": np.array(["ab", "cd", "ef", "gh"]),
                "col_2": np.array(["zx", "one", "two", "three"]),
            },
            "index",
            None,
            ["A", "B", "C", "D"],
        ),
        (
            {
                "index": [("a", "b"), ("a", "c")],
                "columns": [("x", 1), ("y", 2)],
                "data": [np.array([1, 3]), np.array([2, 4])],
                "index_names": ["n1", "n2"],
                "column_names": ["z1", "z2"],
            },
            {
                "index": [("a", "b"), ("a", "c")],
                "columns": [("x", 1), ("y", 2)],
                "data": [cp.array([1, 3]), cp.array([2, 4])],
                "index_names": ["n1", "n2"],
                "column_names": ["z1", "z2"],
            },
            "tight",
            "float64",
            None,
        ),
    ],
)
def test_dataframe_from_dict_cp_np_arrays(
    pd_data, gd_data, orient, dtype, columns
):
    expected = pd.DataFrame.from_dict(
        data=pd_data, orient=orient, dtype=dtype, columns=columns
    )

    actual = cudf.DataFrame.from_dict(
        data=gd_data, orient=orient, dtype=dtype, columns=columns
    )

    assert_eq(expected, actual, check_dtype=dtype is not None)


@pytest.mark.parametrize(
    "data1, data2",
    [(1, 2), (1.0, 2.0), (3, 4.0)],
)
@pytest.mark.parametrize("data3, data4", [(6, 10), (5.0, 9.0), (2, 6.0)])
def test_create_interval_df(data1, data2, data3, data4, interval_closed):
    # df for both pandas and cudf only works when interval is in a list
    expect = pd.DataFrame(
        [pd.Interval(data1, data2, interval_closed)], dtype="interval"
    )
    got = cudf.DataFrame(
        [pd.Interval(data1, data2, interval_closed)], dtype="interval"
    )
    assert_eq(expect, got)

    expect_two = pd.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
        },
        dtype="interval",
    )
    got_two = cudf.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
        },
        dtype="interval",
    )
    assert_eq(expect_two, got_two)

    expect_three = pd.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "c": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
        },
        dtype="interval",
    )

    got_three = cudf.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "c": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
        },
        dtype="interval",
    )
    assert_eq(expect_three, got_three)
