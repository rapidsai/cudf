# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from contextlib import ExitStack as does_not_raise
from sys import getsizeof
from unittest.mock import patch

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import nvstrings

from cudf import concat
from cudf.core import DataFrame, Series
from cudf.core.column.string import StringColumn
from cudf.core.index import StringIndex
from cudf.tests.utils import assert_eq

data_list = [
    ["AbC", "de", "FGHI", "j", "kLm"],
    ["nOPq", None, "RsT", None, "uVw"],
    [None, None, None, None, None],
]

data_id_list = ["no_nulls", "some_nulls", "all_nulls"]

idx_list = [None, [10, 11, 12, 13, 14]]

idx_id_list = ["None_index", "Set_index"]


def raise_builder(flags, exceptions):
    if any(flags):
        return pytest.raises(exceptions)
    else:
        return does_not_raise()


@pytest.fixture(params=data_list, ids=data_id_list)
def data(request):
    return request.param


@pytest.fixture(params=idx_list, ids=idx_id_list)
def index(request):
    return request.param


@pytest.fixture
def ps_gs(data, index):
    ps = pd.Series(data, index=index, dtype="str", name="nice name")
    gs = Series(data, index=index, dtype="str", name="nice name")
    return (ps, gs)


# TODO: Remove this once NVStrings is fully removed / deprecated
@pytest.mark.parametrize("nbytes", [0, 2 ** 10, 2 ** 31 - 1, 2 ** 31, 2 ** 32])
@patch.object(nvstrings.nvstrings, "byte_count")
def test_from_nvstrings_nbytes(mock_byte_count, nbytes):
    import cudf._lib as libcudf

    mock_byte_count.return_value = nbytes
    expectation = raise_builder(
        [nbytes > libcudf.MAX_STRING_COLUMN_BYTES], MemoryError
    )
    with expectation:
        Series(nvstrings.to_device([""]))


@pytest.mark.parametrize("construct", [list, np.array, pd.Series, pa.array])
def test_string_ingest(construct):
    expect = ["a", "a", "b", "c", "a"]
    data = construct(expect)
    got = Series(data)
    assert got.dtype == np.dtype("object")
    assert len(got) == 5
    for idx, val in enumerate(expect):
        assert expect[idx] == got[idx]


def test_string_export(ps_gs):
    ps, gs = ps_gs

    expect = ps
    got = gs.to_pandas()
    pd.testing.assert_series_equal(expect, got)

    expect = np.array(ps)
    got = gs.to_array()
    np.testing.assert_array_equal(expect, got)

    expect = pa.Array.from_pandas(ps)
    got = gs.to_arrow()

    assert pa.Array.equals(expect, got)


@pytest.mark.parametrize(
    "item",
    [
        0,
        2,
        4,
        slice(1, 3),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
        np.array([0, 1, 2, 3, 4]),
        cupy.asarray(np.array([0, 1, 2, 3, 4])),
    ],
)
def test_string_get_item(ps_gs, item):
    ps, gs = ps_gs

    got = gs.iloc[item]
    if isinstance(got, Series):
        got = got.to_arrow()

    if isinstance(item, cupy.ndarray):
        item = cupy.asnumpy(item)

    expect = ps.iloc[item]
    if isinstance(expect, pd.Series):
        expect = pa.Array.from_pandas(expect)
        pa.Array.equals(expect, got)
    else:
        assert expect == got


@pytest.mark.parametrize(
    "item",
    [
        [True] * 5,
        [False] * 5,
        np.array([True] * 5),
        np.array([False] * 5),
        cupy.asarray(np.array([True] * 5)),
        cupy.asarray(np.array([False] * 5)),
        np.random.randint(0, 2, 5).astype("bool").tolist(),
        np.random.randint(0, 2, 5).astype("bool"),
        cupy.asarray(np.random.randint(0, 2, 5).astype("bool")),
    ],
)
def test_string_bool_mask(ps_gs, item):
    ps, gs = ps_gs

    got = gs.iloc[item]
    if isinstance(got, Series):
        got = got.to_arrow()

    if isinstance(item, cupy.ndarray):
        item = cupy.asnumpy(item)

    expect = ps[item]
    if isinstance(expect, pd.Series):
        expect = pa.Array.from_pandas(expect)
        pa.Array.equals(expect, got)
    else:
        assert expect == got


@pytest.mark.parametrize("item", [0, slice(1, 3), slice(5)])
def test_string_repr(ps_gs, item):
    ps, gs = ps_gs

    got_out = gs.iloc[item]
    expect_out = ps.iloc[item]

    expect = str(expect_out)
    got = str(got_out)

    # if isinstance(expect_out, pd.Series):
    #     expect = expect.replace("object", "str")

    assert expect == got


@pytest.mark.parametrize(
    "dtype",
    [
        "str",
        "object",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "bool",
        "datetime64[ms]",
    ],
)
def test_string_astype(dtype):
    if dtype.startswith("int"):
        data = ["1", "2", "3", "4", "5"]
    elif dtype.startswith("float"):
        data = ["1.0", "2.0", "3.0", "4.0", "5.0"]
    elif dtype.startswith("bool"):
        data = ["True", "False", "True", "False", "False"]
    elif dtype.startswith("datetime64"):
        data = [
            "2019-06-04T00:00:00Z",
            "2019-06-04T12:12:12Z",
            "2019-06-03T00:00:00Z",
            "2019-05-04T00:00:00Z",
            "2018-06-04T00:00:00Z",
        ]
    elif dtype == "str" or dtype == "object":
        data = ["ab", "cd", "ef", "gh", "ij"]
    ps = pd.Series(data)
    gs = Series(data)

    # Pandas str --> bool typecasting always returns True if there's a string
    if dtype.startswith("bool"):
        expect = ps == "True"
    else:
        expect = ps.astype(dtype)
    got = gs.astype(dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "str",
        "object",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "bool",
        "datetime64[ms]",
    ],
)
def test_string_empty_astype(dtype):
    data = []
    ps = pd.Series(data, dtype="str")
    gs = Series(data, dtype="str")

    expect = ps.astype(dtype)
    got = gs.astype(dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "bool",
        "datetime64[ms]",
    ],
)
def test_string_numeric_astype(dtype):
    if dtype.startswith("bool"):
        data = [1, 0, 1, 0, 1]
    elif dtype.startswith("int"):
        data = [1, 2, 3, 4, 5]
    elif dtype.startswith("float"):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
    elif dtype.startswith("datetime64"):
        data = [1000000000, 2000000000, 3000000000, 4000000000, 5000000000]
    if dtype.startswith("datetime64"):
        ps = pd.Series(data, dtype="datetime64[ns]")
        gs = Series.from_pandas(ps)
    else:
        ps = pd.Series(data, dtype=dtype)
        gs = Series(data, dtype=dtype)

    # Pandas datetime64 --> str typecasting returns arbitrary format depending
    # on the data, so making it consistent unless we choose to match the
    # behavior
    if dtype.startswith("datetime64"):
        expect = ps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        expect = ps.astype("str")
    got = gs.astype("str")

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "bool",
        "datetime64[ms]",
    ],
)
def test_string_empty_numeric_astype(dtype):
    data = []

    if dtype.startswith("datetime64"):
        ps = pd.Series(data, dtype="datetime64[ns]")
    else:
        ps = pd.Series(data, dtype=dtype)
    gs = Series(data, dtype=dtype)

    expect = ps.astype("str")
    got = gs.astype("str")

    assert_eq(expect, got)


def test_string_concat():
    data1 = ["a", "b", "c", "d", "e"]
    data2 = ["f", "g", "h", "i", "j"]

    ps1 = pd.Series(data1)
    ps2 = pd.Series(data2)
    gs1 = Series(data1)
    gs2 = Series(data2)

    expect = pd.concat([ps1, ps2])
    got = concat([gs1, gs2])

    assert_eq(expect, got)


@pytest.mark.parametrize("ascending", [True, False])
def test_string_sort(ps_gs, ascending):
    ps, gs = ps_gs

    expect = ps.sort_values(ascending=ascending)
    got = gs.sort_values(ascending=ascending)

    assert_eq(expect, got)


def test_string_len(ps_gs):
    ps, gs = ps_gs

    expect = ps.str.len()
    got = gs.str.len()

    # Can't handle nulls in Pandas so use PyArrow instead
    # Pandas will return as a float64 so need to typecast to int32
    expect = pa.array(expect, from_pandas=True).cast(pa.int32())
    got = got.to_arrow()
    assert pa.Array.equals(expect, got)


@pytest.mark.parametrize(
    "others",
    [
        None,
        ["f", "g", "h", "i", "j"],
        ("f", "g", "h", "i", "j"),
        pd.Series(["f", "g", "h", "i", "j"]),
        pd.Index(["f", "g", "h", "i", "j"]),
        (["f", "g", "h", "i", "j"], ["f", "g", "h", "i", "j"]),
        [["f", "g", "h", "i", "j"], ["f", "g", "h", "i", "j"]],
        (
            pd.Series(["f", "g", "h", "i", "j"]),
            ["f", "a", "b", "f", "a"],
            pd.Series(["f", "g", "h", "i", "j"]),
            ["f", "a", "b", "f", "a"],
            ["f", "a", "b", "f", "a"],
            pd.Index(["1", "2", "3", "4", "5"]),
            ["f", "a", "b", "f", "a"],
            pd.Index(["f", "g", "h", "i", "j"]),
        ),
        [
            pd.Index(["f", "g", "h", "i", "j"]),
            ["f", "a", "b", "f", "a"],
            pd.Series(["f", "g", "h", "i", "j"]),
            ["f", "a", "b", "f", "a"],
            ["f", "a", "b", "f", "a"],
            pd.Index(["f", "g", "h", "i", "j"]),
            ["f", "a", "b", "f", "a"],
            pd.Index(["f", "g", "h", "i", "j"]),
        ],
    ],
)
@pytest.mark.parametrize("sep", [None, "", " ", "|", ",", "|||"])
@pytest.mark.parametrize("na_rep", [None, "", "null", "a"])
@pytest.mark.parametrize(
    "index",
    [
        ["1", "2", "3", "4", "5"],
        pd.Series(["1", "2", "3", "4", "5"]),
        pd.Index(["1", "2", "3", "4", "5"]),
    ],
)
def test_string_cat(ps_gs, others, sep, na_rep, index):
    ps, gs = ps_gs

    pd_others = others
    if isinstance(pd_others, pd.Series):
        pd_others = pd_others.values
    expect = ps.str.cat(others=pd_others, sep=sep, na_rep=na_rep)
    got = gs.str.cat(others=others, sep=sep, na_rep=na_rep)

    assert_eq(expect, got)

    ps.index = index
    gs.index = index

    expect = ps.str.cat(others=ps.index, sep=sep, na_rep=na_rep)
    got = gs.str.cat(others=gs.index, sep=sep, na_rep=na_rep)

    assert_eq(expect, got)

    expect = ps.str.cat(others=[ps.index] + [ps.index], sep=sep, na_rep=na_rep)
    got = gs.str.cat(others=[gs.index] + [gs.index], sep=sep, na_rep=na_rep)

    assert_eq(expect, got)

    expect = ps.str.cat(others=(ps.index, ps.index), sep=sep, na_rep=na_rep)
    got = gs.str.cat(others=(gs.index, gs.index), sep=sep, na_rep=na_rep)

    assert_eq(expect, got)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("sep", [None, "", " ", "|", ",", "|||"])
@pytest.mark.parametrize("na_rep", [None, "", "null", "a"])
def test_string_cat_str(ps_gs, sep, na_rep):
    ps, gs = ps_gs

    got = gs.str.cat(gs.str, sep=sep, na_rep=na_rep)
    expect = ps.str.cat(ps.str, sep=sep, na_rep=na_rep)

    assert_eq(expect, got)


@pytest.mark.xfail(raises=(NotImplementedError, AttributeError))
@pytest.mark.parametrize("sep", [None, "", " ", "|", ",", "|||"])
def test_string_join(ps_gs, sep):
    ps, gs = ps_gs

    expect = ps.str.join(sep)
    got = gs.str.join(sep)

    assert_eq(expect, got)


@pytest.mark.parametrize("pat", [r"(a)", r"(f)", r"([a-z])", r"([A-Z])"])
@pytest.mark.parametrize("expand", [True, False])
@pytest.mark.parametrize("flags,flags_raise", [(0, 0), (1, 1)])
def test_string_extract(ps_gs, pat, expand, flags, flags_raise):
    ps, gs = ps_gs
    expectation = raise_builder([flags_raise], NotImplementedError)

    with expectation:
        expect = ps.str.extract(pat, flags=flags, expand=expand)
        got = gs.str.extract(pat, flags=flags, expand=expand)

        assert_eq(expect, got)


@pytest.mark.parametrize(
    "pat,regex",
    [
        ("a", False),
        ("a", True),
        ("f", False),
        (r"[a-z]", True),
        (r"[A-Z]", True),
        ("hello", False),
        ("FGHI", False),
    ],
)
@pytest.mark.parametrize("flags,flags_raise", [(0, 0), (1, 1)])
@pytest.mark.parametrize("na,na_raise", [(np.nan, 0), (None, 1), ("", 1)])
def test_string_contains(ps_gs, pat, regex, flags, flags_raise, na, na_raise):
    ps, gs = ps_gs

    expectation = raise_builder([flags_raise, na_raise], NotImplementedError)

    with expectation:
        expect = ps.str.contains(pat, flags=flags, na=na, regex=regex)
        got = gs.str.contains(pat, flags=flags, na=na, regex=regex)
        assert_eq(expect, got)


# Pandas isn't respect the `n` parameter so ignoring it in test parameters
@pytest.mark.parametrize(
    "pat,regex",
    [("a", False), ("f", False), (r"[a-z]", True), (r"[A-Z]", True)],
)
@pytest.mark.parametrize("repl", ["qwerty", "", " "])
@pytest.mark.parametrize("case,case_raise", [(None, 0), (True, 1), (False, 1)])
@pytest.mark.parametrize("flags,flags_raise", [(0, 0), (1, 1)])
def test_string_replace(
    ps_gs, pat, repl, case, case_raise, flags, flags_raise, regex
):
    ps, gs = ps_gs

    expectation = raise_builder([case_raise, flags_raise], NotImplementedError)

    with expectation:
        expect = ps.str.replace(pat, repl, case=case, flags=flags, regex=regex)
        got = gs.str.replace(pat, repl, case=case, flags=flags, regex=regex)

        assert_eq(expect, got)


def test_string_lower(ps_gs):
    ps, gs = ps_gs

    expect = ps.str.lower()
    got = gs.str.lower()

    assert_eq(expect, got)


def test_string_upper(ps_gs):
    ps, gs = ps_gs

    expect = ps.str.upper()
    got = gs.str.upper()

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        ["a b", " c ", "   d", "e   ", "f"],
        ["a-b", "-c-", "---d", "e---", "f"],
        ["ab", "c", "d", "e", "f"],
        [None, None, None, None, None],
    ],
)
@pytest.mark.parametrize("pat", [None, " ", "-"])
@pytest.mark.parametrize("n", [-1, 0, 1, 3, 10])
@pytest.mark.parametrize("expand,expand_raise", [(True, 0), (False, 1)])
def test_string_split(data, pat, n, expand, expand_raise):

    if data in (["a b", " c ", "   d", "e   ", "f"],) and pat is None:
        pytest.xfail("None pattern split algorithm not implemented yet")

    ps = pd.Series(data, dtype="str")
    gs = Series(data, dtype="str")

    expectation = raise_builder([expand_raise], NotImplementedError)

    with expectation:
        expect = ps.str.split(pat=pat, n=n, expand=expand)
        got = gs.str.split(pat=pat, n=n, expand=expand)

        assert_eq(expect, got)


@pytest.mark.parametrize(
    "str_data,str_data_raise",
    [
        ([], 0),
        (["a", "b", "c", "d", "e"], 0),
        ([None, None, None, None, None], 1),
    ],
)
@pytest.mark.parametrize("num_keys", [1, 2, 3])
@pytest.mark.parametrize(
    "how,how_raise", [("left", 0), ("right", 1), ("inner", 0), ("outer", 0)]
)
def test_string_join_key(str_data, str_data_raise, num_keys, how, how_raise):
    other_data = [1, 2, 3, 4, 5][: len(str_data)]

    pdf = pd.DataFrame()
    gdf = DataFrame()
    for i in range(num_keys):
        pdf[i] = pd.Series(str_data, dtype="str")
        gdf[i] = Series(str_data, dtype="str")
    pdf["a"] = other_data
    gdf["a"] = other_data

    pdf2 = pdf.copy()
    gdf2 = gdf.copy()

    expectation = raise_builder(
        [how_raise, str_data_raise], (NotImplementedError, AssertionError)
    )

    with expectation:
        expect = pdf.merge(pdf2, on=list(range(num_keys)), how=how)
        got = gdf.merge(gdf2, on=list(range(num_keys)), how=how)

        if len(expect) == 0 and len(got) == 0:
            expect = expect.reset_index(drop=True)
            got = got[expect.columns]

        assert_eq(expect, got)


@pytest.mark.parametrize(
    "str_data_nulls",
    [
        ["a", "b", "c"],
        ["a", "b", "f", "g"],
        ["f", "g", "h", "i", "j"],
        ["f", "g", "h"],
        [None, None, None, None, None],
        [],
    ],
)
def test_string_join_key_nulls(str_data_nulls):
    str_data = ["a", "b", "c", "d", "e"]
    other_data = [1, 2, 3, 4, 5]

    other_data_nulls = [6, 7, 8, 9, 10][: len(str_data_nulls)]

    pdf = pd.DataFrame()
    gdf = DataFrame()
    pdf["key"] = pd.Series(str_data, dtype="str")
    gdf["key"] = Series(str_data, dtype="str")
    pdf["vals"] = other_data
    gdf["vals"] = other_data

    pdf2 = pd.DataFrame()
    gdf2 = DataFrame()
    pdf2["key"] = pd.Series(str_data_nulls, dtype="str")
    gdf2["key"] = Series(str_data_nulls, dtype="str")
    pdf2["vals"] = pd.Series(other_data_nulls, dtype="int64")
    gdf2["vals"] = Series(other_data_nulls, dtype="int64")

    expect = pdf.merge(pdf2, on="key", how="left")
    got = gdf.merge(gdf2, on="key", how="left")

    if len(expect) == 0 and len(got) == 0:
        expect = expect.reset_index(drop=True)
        got = got[expect.columns]

    expect["vals_y"] = expect["vals_y"].fillna(-1).astype("int64")

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "str_data", [[], ["a", "b", "c", "d", "e"], [None, None, None, None, None]]
)
@pytest.mark.parametrize("num_cols", [1, 2, 3])
@pytest.mark.parametrize(
    "how,how_raise", [("left", 0), ("right", 1), ("inner", 0), ("outer", 0)]
)
def test_string_join_non_key(str_data, num_cols, how, how_raise):
    other_data = [1, 2, 3, 4, 5][: len(str_data)]

    pdf = pd.DataFrame()
    gdf = DataFrame()
    for i in range(num_cols):
        pdf[i] = pd.Series(str_data, dtype="str")
        gdf[i] = Series(str_data, dtype="str")
    pdf["a"] = other_data
    gdf["a"] = other_data

    pdf2 = pdf.copy()
    gdf2 = gdf.copy()

    expectation = raise_builder([how_raise], NotImplementedError)

    with expectation:
        expect = pdf.merge(pdf2, on=["a"], how=how)
        got = gdf.merge(gdf2, on=["a"], how=how)

        if len(expect) == 0 and len(got) == 0:
            expect = expect.reset_index(drop=True)
            got = got[expect.columns]

        assert_eq(expect, got)


@pytest.mark.parametrize(
    "str_data_nulls",
    [
        ["a", "b", "c"],
        ["a", "b", "f", "g"],
        ["f", "g", "h", "i", "j"],
        ["f", "g", "h"],
        [None, None, None, None, None],
        [],
    ],
)
def test_string_join_non_key_nulls(str_data_nulls):
    str_data = ["a", "b", "c", "d", "e"]
    other_data = [1, 2, 3, 4, 5]

    other_data_nulls = [6, 7, 8, 9, 10][: len(str_data_nulls)]

    pdf = pd.DataFrame()
    gdf = DataFrame()
    pdf["vals"] = pd.Series(str_data, dtype="str")
    gdf["vals"] = Series(str_data, dtype="str")
    pdf["key"] = other_data
    gdf["key"] = other_data

    pdf2 = pd.DataFrame()
    gdf2 = DataFrame()
    pdf2["vals"] = pd.Series(str_data_nulls, dtype="str")
    gdf2["vals"] = Series(str_data_nulls, dtype="str")
    pdf2["key"] = pd.Series(other_data_nulls, dtype="int64")
    gdf2["key"] = Series(other_data_nulls, dtype="int64")

    expect = pdf.merge(pdf2, on="key", how="left")
    got = gdf.merge(gdf2, on="key", how="left")

    if len(expect) == 0 and len(got) == 0:
        expect = expect.reset_index(drop=True)
        got = got[expect.columns]

    assert_eq(expect, got)


def test_string_join_values_nulls():
    left_dict = [
        {"b": "MATCH 1", "a": 1.0},
        {"b": "MATCH 1", "a": 1.0},
        {"b": "LEFT NO MATCH 1", "a": -1.0},
        {"b": "MATCH 2", "a": 2.0},
        {"b": "MATCH 2", "a": 2.0},
        {"b": "MATCH 1", "a": 1.0},
        {"b": "MATCH 1", "a": 1.0},
        {"b": "MATCH 2", "a": 2.0},
        {"b": "MATCH 2", "a": 2.0},
        {"b": "LEFT NO MATCH 2", "a": -2.0},
        {"b": "MATCH 3", "a": 3.0},
        {"b": "MATCH 3", "a": 3.0},
    ]

    right_dict = [
        {"b": "RIGHT NO MATCH 1", "c": -1.0},
        {"b": "MATCH 3", "c": 3.0},
        {"b": "MATCH 2", "c": 2.0},
        {"b": "RIGHT NO MATCH 2", "c": -2.0},
        {"b": "RIGHT NO MATCH 3", "c": -3.0},
        {"b": "MATCH 1", "c": 1.0},
    ]

    left_pdf = pd.DataFrame(left_dict)
    right_pdf = pd.DataFrame(right_dict)

    left_gdf = DataFrame.from_pandas(left_pdf)
    right_gdf = DataFrame.from_pandas(right_pdf)

    expect = left_pdf.merge(right_pdf, how="left", on="b")
    got = left_gdf.merge(right_gdf, how="left", on="b")

    expect = expect.sort_values(by=["a", "b", "c"]).reset_index(drop=True)
    got = got.sort_values(by=["a", "b", "c"]).reset_index(drop=True)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "str_data", [[], ["a", "b", "c", "d", "e"], [None, None, None, None, None]]
)
@pytest.mark.parametrize("num_keys", [1, 2, 3])
def test_string_groupby_key(str_data, num_keys):
    other_data = [1, 2, 3, 4, 5][: len(str_data)]

    pdf = pd.DataFrame()
    gdf = DataFrame()
    for i in range(num_keys):
        pdf[i] = pd.Series(str_data, dtype="str")
        gdf[i] = Series(str_data, dtype="str")
    pdf["a"] = other_data
    gdf["a"] = other_data

    expect = pdf.groupby(list(range(num_keys)), as_index=False).count()
    got = gdf.groupby(list(range(num_keys)), as_index=False).count()

    expect = expect.sort_values([0]).reset_index(drop=True)
    got = got.sort_values([0]).reset_index(drop=True)

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "str_data", [[], ["a", "b", "c", "d", "e"], [None, None, None, None, None]]
)
@pytest.mark.parametrize("num_cols", [1, 2, 3])
@pytest.mark.parametrize("agg", ["count", "max", "min"])
def test_string_groupby_non_key(str_data, num_cols, agg):
    other_data = [1, 2, 3, 4, 5][: len(str_data)]

    pdf = pd.DataFrame()
    gdf = DataFrame()
    for i in range(num_cols):
        pdf[i] = pd.Series(str_data, dtype="str")
        gdf[i] = Series(str_data, dtype="str")
    pdf["a"] = other_data
    gdf["a"] = other_data

    expect = getattr(pdf.groupby("a", as_index=False), agg)()
    got = getattr(gdf.groupby("a", as_index=False), agg)()

    expect = expect.sort_values(["a"]).reset_index(drop=True)
    got = got.sort_values(["a"]).reset_index(drop=True)

    if agg in ["min", "max"] and len(expect) == 0 and len(got) == 0:
        for i in range(num_cols):
            expect[i] = expect[i].astype("str")

    assert_eq(expect, got, check_dtype=False)


def test_string_groupby_key_index():
    str_data = ["a", "b", "c", "d", "e"]
    other_data = [1, 2, 3, 4, 5]

    pdf = pd.DataFrame()
    gdf = DataFrame()
    pdf["a"] = pd.Series(str_data, dtype="str")
    gdf["a"] = Series(str_data, dtype="str")
    pdf["b"] = other_data
    gdf["b"] = other_data

    expect = pdf.groupby("a").count()
    got = gdf.groupby("a").count()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("scalar", ["a", None])
def test_string_set_scalar(scalar):
    pdf = pd.DataFrame()
    pdf["a"] = [1, 2, 3, 4, 5]
    gdf = DataFrame.from_pandas(pdf)

    pdf["b"] = "a"
    gdf["b"] = "a"

    assert_eq(pdf["b"], gdf["b"])
    assert_eq(pdf, gdf)


def test_string_index():
    from cudf.core.column import as_column
    from cudf.core.index import as_index

    pdf = pd.DataFrame(np.random.rand(5, 5))
    gdf = DataFrame.from_pandas(pdf)
    stringIndex = ["a", "b", "c", "d", "e"]
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = np.array(["a", "b", "c", "d", "e"])
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = StringIndex(["a", "b", "c", "d", "e"], name="name")
    pdf.index = stringIndex.to_pandas()
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = as_index(as_column(["a", "b", "c", "d", "e"]), name="name")
    pdf.index = stringIndex.to_pandas()
    gdf.index = stringIndex
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "item",
    [
        ["Cbe", "cbe", "CbeD", "Cb", "ghi", "Cb"],
        ["a", "a", "a", "a", "A"],
        ["A"],
        ["abc", "xyz", None, "ab", "123"],
        [None, None, "abc", None, "abc"],
    ],
)
def test_string_unique(item):
    ps = pd.Series(item)
    gs = Series(item)
    # Pandas `unique` returns a numpy array
    pres = pd.Series(ps.unique())
    # Nvstrings returns sorted unique with `None` placed before other strings
    pres = pres.sort_values(na_position="first").reset_index(drop=True)
    gres = gs.unique()
    assert_eq(pres, gres)


def test_string_slice():
    df = DataFrame({"a": ["hello", "world"]})
    pdf = pd.DataFrame({"a": ["hello", "world"]})
    a_slice_got = df.a.str.slice(0, 2)
    a_slice_expected = pdf.a.str.slice(0, 2)

    assert isinstance(a_slice_got, Series)
    assert_eq(a_slice_expected, a_slice_got)


def test_string_equality():
    data1 = ["b", "c", "d", "a", "c"]
    data2 = ["a", None, "c", "a", "c"]

    ps1 = pd.Series(data1)
    ps2 = pd.Series(data2)
    gs1 = Series(data1)
    gs2 = Series(data2)

    expect = ps1 == ps2
    got = gs1 == gs2

    assert_eq(expect, got.fillna(False))

    expect = ps1 == "m"
    got = gs1 == "m"

    assert_eq(expect, got.fillna(False))

    ps1 = pd.Series(["a"])
    gs1 = Series(["a"])

    expect = ps1 == "m"
    got = gs1 == "m"

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "lhs",
    [
        ["Cbe", "cbe", "CbeD", "Cb", "ghi", "Cb"],
        ["abc", "xyz", "a", "ab", "123", "097"],
    ],
)
@pytest.mark.parametrize(
    "rhs",
    [
        ["Cbe", "cbe", "CbeD", "Cb", "ghi", "Cb"],
        ["a", "a", "a", "a", "A", "z"],
    ],
)
def test_string_binary_op_add(lhs, rhs):
    pds = pd.Series(lhs) + pd.Series(rhs)
    gds = Series(lhs) + Series(rhs)

    assert_eq(pds, gds)


@pytest.mark.parametrize("name", [None, "new name", 123])
def test_string_misc_name(ps_gs, name):
    ps, gs = ps_gs
    ps.name = name
    gs.name = name

    expect = ps.str.slice(0, 1)
    got = gs.str.slice(0, 1)

    assert_eq(expect, got)
    assert_eq(ps + ps, gs + gs)
    assert_eq(ps + "RAPIDS", gs + "RAPIDS")
    assert_eq("RAPIDS" + ps, "RAPIDS" + gs)


def test_string_no_children_properties():
    empty_col = StringColumn(children=())
    assert empty_col.base_children == ()
    assert empty_col.base_size == 0

    assert empty_col.children == ()
    assert empty_col.size == 0

    assert empty_col._nbytes == 0
    assert getsizeof(empty_col) >= 0  # Accounts for Python GC overhead


@pytest.mark.parametrize(
    "string",
    [
        ["Cbe", "cbe", "CbeD", "Cb", "ghi", "Cb"],
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""],
    ],
)
@pytest.mark.parametrize(
    "index", [-100, -5, -2, -6, -1, 0, 1, 2, 3, 9, 10, 100]
)
def test_string_get(string, index):
    pds = pd.Series(string)
    gds = Series(string)

    assert_eq(
        pds.str.get(index).fillna(""), gds.str.get(index).fillna(""),
    )


@pytest.mark.parametrize(
    "string",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""],
        ["koala", "fox", "chameleon"],
    ],
)
@pytest.mark.parametrize(
    "number", [-10, 0, 1, 3, 10],
)
@pytest.mark.parametrize(
    "diff", [0, 2, 5, 9],
)
def test_string_slice_str(string, number, diff):
    pds = pd.Series(string)
    gds = Series(string)

    assert_eq(pds.str.slice(start=number), gds.str.slice(start=number))
    assert_eq(pds.str.slice(stop=number), gds.str.slice(stop=number))
    assert_eq(pds.str.slice(), gds.str.slice())
    assert_eq(
        pds.str.slice(start=number, stop=number + diff),
        gds.str.slice(start=number, stop=number + diff),
    )
    if diff != 0:
        assert_eq(pds.str.slice(step=diff), gds.str.slice(step=diff))
        assert_eq(
            pds.str.slice(start=number, stop=number + diff, step=diff),
            gds.str.slice(start=number, stop=number + diff, step=diff),
        )


def test_string_slice_from():
    gs = Series(["hello world", "holy accéntéd", "batman", None, ""])
    d_starts = Series([2, 3, 0, -1, -1], dtype=np.int32)
    d_stops = Series([-1, -1, 0, -1, -1], dtype=np.int32)
    got = gs.str.slice_from(starts=d_starts._column, stops=d_stops._column)
    expected = Series(["llo world", "y accéntéd", "", None, ""])
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "string",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""],
        ["koala", "fox", "chameleon"],
    ],
)
@pytest.mark.parametrize("number", [0, 1, 10])
@pytest.mark.parametrize("diff", [0, 2, 9])
@pytest.mark.parametrize("repr", ["2", "!!"])
def test_string_slice_replace(string, number, diff, repr):
    pds = pd.Series(string)
    gds = Series(string)

    assert_eq(
        pds.str.slice_replace(start=number, repl=repr),
        gds.str.slice_replace(start=number, repl=repr),
        check_dtype=False,
    )
    assert_eq(
        pds.str.slice_replace(stop=number, repl=repr),
        gds.str.slice_replace(stop=number, repl=repr),
    )
    assert_eq(pds.str.slice_replace(), gds.str.slice_replace())
    assert_eq(
        pds.str.slice_replace(start=number, stop=number + diff),
        gds.str.slice_replace(start=number, stop=number + diff),
    )
    assert_eq(
        pds.str.slice_replace(start=number, stop=number + diff, repl=repr),
        gds.str.slice_replace(start=number, stop=number + diff, repl=repr),
        check_dtype=False,
    )


def test_string_insert():
    gs = Series(["hello world", "holy accéntéd", "batman", None, ""])

    ps = pd.Series(["hello world", "holy accéntéd", "batman", None, ""])

    assert_eq(gs.str.insert(0, ""), gs)
    assert_eq(gs.str.insert(0, "+"), "+" + ps)
    assert_eq(gs.str.insert(-1, "---"), ps + "---")
    assert_eq(
        gs.str.insert(5, "---"),
        ps.str.slice(stop=5) + "---" + ps.str.slice(start=5),
    )


_string_char_types_data = [
    ["abc", "xyz", "a", "ab", "123", "097"],
    ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""],
    ["koala", "fox", "chameleon"],
    [
        "1234567890",
        "de",
        "1.75",
        "-34",
        "+9.8",
        "7¼",
        "x³",
        "2³",
        "12⅝",
        "",
        "\t\r\n ",
    ],
    ["one", "one1", "1", ""],
    ["A B", "1.5", "3,000"],
    ["23", "³", "⅕", ""],
    [" ", "\t\r\n ", ""],
    ["leopard", "Golden Eagle", "SNAKE", ""],
    [r"¯\_(ツ)_/¯", "(╯°□°)╯︵ ┻━┻", "┬─┬ノ( º _ ºノ)"],
    ["a1", "A1", "a!", "A!", "!1", "aA"],
]


@pytest.mark.parametrize(
    "type_op",
    [
        "isdecimal",
        "isalnum",
        "isalpha",
        "isdigit",
        "isnumeric",
        "isupper",
        "islower",
    ],
)
@pytest.mark.parametrize("data", _string_char_types_data)
def test_string_char_types(type_op, data):
    gs = Series(data)
    ps = pd.Series(data)

    assert_eq(getattr(gs.str, type_op)(), getattr(ps.str, type_op)())


@pytest.mark.parametrize(
    "case_op", ["title", "capitalize", "lower", "upper", "swapcase"]
)
@pytest.mark.parametrize(
    "data",
    [
        *_string_char_types_data,
        [
            None,
            "The quick bRoWn fox juMps over the laze DOG",
            '123nr98nv9rev!$#INF4390v03n1243<>?}{:-"',
            "accénted",
        ],
    ],
)
def test_string_char_case(case_op, data):
    gs = Series(data)
    ps = pd.Series(data)

    s = gs.str
    a = getattr(s, case_op)

    assert_eq(a(), getattr(ps.str, case_op)())

    assert_eq(gs.str.capitalize(), ps.str.capitalize())
    assert_eq(gs.str.isdecimal(), ps.str.isdecimal())
    assert_eq(gs.str.isalnum(), ps.str.isalnum())
    assert_eq(gs.str.isalpha(), ps.str.isalpha())
    assert_eq(gs.str.isdigit(), ps.str.isdigit())
    assert_eq(gs.str.isnumeric(), ps.str.isnumeric())
    assert_eq(gs.str.isspace(), ps.str.isspace())

    assert_eq(gs.str.isempty(), ps == "")


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
    ],
)
def test_strings_rpartition(data):
    gs = Series(data)
    ps = pd.Series(data)

    assert_eq(ps.str.rpartition(), gs.str.rpartition())
    assert_eq(ps.str.rpartition("-"), gs.str.rpartition("-"))
    assert_eq(ps.str.rpartition(","), gs.str.rpartition(","))


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
    ],
)
def test_strings_partition(data):
    gs = Series(data)
    ps = pd.Series(data)

    assert_eq(ps.str.partition(), gs.str.partition())
    assert_eq(ps.str.partition(","), gs.str.partition(","))
    assert_eq(ps.str.partition("-"), gs.str.partition("-"))


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        [
            "this is a regular sentence",
            "https://docs.python.org/3/tutorial/index.html",
            None,
        ],
    ],
)
@pytest.mark.parametrize("n", [-1, 2, 1, 9])
@pytest.mark.parametrize("expand", [True])
def test_strings_rsplit(data, n, expand):
    gs = Series(data)
    ps = pd.Series(data)

    pd.testing.assert_frame_equal(
        ps.str.rsplit(n=n, expand=expand).reset_index(),
        gs.str.rsplit(n=n, expand=expand).to_pandas().reset_index(),
        check_index_type=False,
    )
    assert_eq(
        ps.str.rsplit(",", n=n, expand=expand),
        gs.str.rsplit(",", n=n, expand=expand),
    )
    assert_eq(
        ps.str.rsplit("-", n=n, expand=expand),
        gs.str.rsplit("-", n=n, expand=expand),
    )


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        [
            "this is a regular sentence",
            "https://docs.python.org/3/tutorial/index.html",
            None,
        ],
    ],
)
@pytest.mark.parametrize("n", [-1, 2, 1, 9])
@pytest.mark.parametrize("expand", [True])
def test_strings_split(data, n, expand):
    gs = Series(data)
    ps = pd.Series(data)

    pd.testing.assert_frame_equal(
        ps.str.split(n=n, expand=expand).reset_index(),
        gs.str.split(n=n, expand=expand).to_pandas().reset_index(),
        check_index_type=False,
    )

    assert_eq(
        ps.str.split(",", n=n, expand=expand),
        gs.str.split(",", n=n, expand=expand),
    )
    assert_eq(
        ps.str.split("-", n=n, expand=expand),
        gs.str.split("-", n=n, expand=expand),
    )


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        [
            "this is a regular sentence",
            "https://docs.python.org/3/tutorial/index.html",
            None,
        ],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize(
    "to_strip", ["⅕", None, "123.", ".!? \n\t", "123.!? \n\t", " ", ".", ","]
)
def test_strings_strip_tests(data, to_strip):
    gs = Series(data)
    ps = pd.Series(data)

    assert_eq(ps.str.strip(to_strip=to_strip), gs.str.strip(to_strip=to_strip))
    assert_eq(
        ps.str.rstrip(to_strip=to_strip), gs.str.rstrip(to_strip=to_strip)
    )
    assert_eq(
        ps.str.lstrip(to_strip=to_strip), gs.str.lstrip(to_strip=to_strip)
    )


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        [
            "this is a regular sentence",
            "https://docs.python.org/3/tutorial/index.html",
            None,
        ],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize("width", [0, 1, 4, 9, 100])
@pytest.mark.parametrize("fillchar", ["⅕", "1", ".", "t", " ", ","])
def test_strings_filling_tests(data, width, fillchar):
    gs = Series(data)
    ps = pd.Series(data)

    assert_eq(
        ps.str.center(width=width, fillchar=fillchar),
        gs.str.center(width=width, fillchar=fillchar),
    )
    assert_eq(
        ps.str.ljust(width=width, fillchar=fillchar),
        gs.str.ljust(width=width, fillchar=fillchar),
    )
    assert_eq(
        ps.str.rjust(width=width, fillchar=fillchar),
        gs.str.rjust(width=width, fillchar=fillchar),
    )


@pytest.mark.parametrize(
    "data",
    [
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["+23", "³", "⅕", ""],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        [" ", "\t\r\n ", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize("width", [0, 1, 4, 6, 9, 100])
def test_strings_zfill_tests(data, width):
    gs = Series(data)
    ps = pd.Series(data)

    assert_eq(ps.str.zfill(width=width), gs.str.zfill(width=width))


@pytest.mark.parametrize(
    "data",
    [
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["+23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize("width", [0, 1, 4, 9, 100])
@pytest.mark.parametrize(
    "side", ["left", "right", "both"],
)
@pytest.mark.parametrize("fillchar", [" ", ".", "\n", "+", "\t"])
def test_strings_pad_tests(data, width, side, fillchar):
    gs = Series(data)
    ps = pd.Series(data)

    assert_eq(
        ps.str.pad(width=width, side=side, fillchar=fillchar),
        gs.str.pad(width=width, side=side, fillchar=fillchar),
    )


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        # [" ", "\t\r\n ", ""],
        ["leopard", "Golden Eagle", "SNAKE", ""],
        ["line to be wrapped", "another line to be wrapped"],
    ],
)
@pytest.mark.parametrize("width", [1, 4, 8, 12, 100])
def test_string_wrap(data, width):
    gs = Series(data)
    ps = pd.Series(data)

    assert_eq(
        gs.str.wrap(width=width),
        ps.str.wrap(
            width=width,
            break_long_words=False,
            expand_tabs=False,
            replace_whitespace=True,
            drop_whitespace=True,
            break_on_hyphens=False,
        ),
    )


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
        ["line to be wrapped", "another line to be wrapped"],
    ],
)
@pytest.mark.parametrize("pat", ["a", " ", "\t", "another", "0", r"\$"])
def test_string_count(data, pat):
    gs = Series(data)
    ps = pd.Series(data)

    assert_eq(gs.str.count(pat=pat), ps.str.count(pat=pat), check_dtype=False)


def test_string_findall():
    ps = pd.Series(["Lion", "Monkey", "Rabbit"])
    gs = Series(["Lion", "Monkey", "Rabbit"])

    assert_eq(ps.str.findall("Monkey")[1][0], gs.str.findall("Monkey")[0][1])
    assert_eq(ps.str.findall("on")[0][0], gs.str.findall("on")[0][0])
    assert_eq(ps.str.findall("on")[1][0], gs.str.findall("on")[0][1])
    assert_eq(ps.str.findall("on$")[0][0], gs.str.findall("on$")[0][0])
    assert_eq(ps.str.findall("b")[2][1], gs.str.findall("b")[1][2])


def test_string_replace_multi():
    ps = pd.Series(["hello", "goodbye"])
    gs = Series(["hello", "goodbye"])
    expect = ps.str.replace("e", "E").str.replace("o", "O")
    got = gs.str.replace(["e", "o"], ["E", "O"])

    assert_eq(expect, got)

    ps = pd.Series(["foo", "fuz", np.nan])
    gs = Series.from_pandas(ps)

    expect = ps.str.replace("f.", "ba", regex=True)
    got = gs.str.replace(["f."], ["ba"], regex=True)
    assert_eq(expect, got)

    ps = pd.Series(["f.o", "fuz", np.nan])
    gs = Series.from_pandas(ps)

    expect = ps.str.replace("f.", "ba", regex=False)
    got = gs.str.replace(["f."], ["ba"], regex=False)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "find",
    [
        "(\\d)(\\d)",
        "(\\d)(\\d)",
        "(\\d)(\\d)",
        "(\\d)(\\d)",
        "([a-z])-([a-z])",
        "([a-z])-([a-zé])",
        "([a-z])-([a-z])",
        "([a-z])-([a-zé])",
    ],
)
@pytest.mark.parametrize(
    "replace",
    ["\\1-\\2", "V\\2-\\1", "\\1 \\2", "\\2 \\1", "X\\1+\\2Z", "X\\1+\\2Z"],
)
def test_string_replace_with_backrefs(find, replace):
    s = [
        "A543",
        "Z756",
        "",
        None,
        "tést-string",
        "two-thréé four-fivé",
        "abcd-éfgh",
        "tést-string-again",
    ]
    ps = pd.Series(s)
    gs = Series(s)
    got = gs.str.replace_with_backrefs(find, replace)
    expected = ps.str.replace(find, replace, regex=True)
    assert_eq(got, expected)


def test_string_table_view_creation():
    data = ["hi"] * 25 + [None] * 2027
    psr = pd.Series(data)
    gsr = Series.from_pandas(psr)

    expect = psr[:1]
    got = gsr[:1]

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
        ["line to be wrapped", "another line to be wrapped"],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize(
    "pat", ["", None, " ", "a", "abc", "cat", "$", "\n"],
)
def test_string_starts_ends(data, pat):
    ps = pd.Series(data)
    gs = Series(data)

    assert_eq(
        ps.str.startswith(pat), gs.str.startswith(pat), check_dtype=False
    )
    assert_eq(ps.str.endswith(pat), gs.str.endswith(pat), check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
        ["line to be wrapped", "another line to be wrapped"],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize(
    "sub", ["", " ", "a", "abc", "cat", "$", "\n"],
)
def test_string_find(data, sub):
    ps = pd.Series(data)
    gs = Series(data)

    assert_eq(ps.str.find(sub).fillna(-1), gs.str.find(sub), check_dtype=False)
    assert_eq(
        ps.str.find(sub, start=1).fillna(-1),
        gs.str.find(sub, start=1),
        check_dtype=False,
    )
    assert_eq(
        ps.str.find(sub, end=10).fillna(-1),
        gs.str.find(sub, end=10),
        check_dtype=False,
    )
    assert_eq(
        ps.str.find(sub, start=2, end=10).fillna(-1),
        gs.str.find(sub, start=2, end=10),
        check_dtype=False,
    )

    assert_eq(
        ps.str.rfind(sub).fillna(-1), gs.str.rfind(sub), check_dtype=False
    )
    assert_eq(
        ps.str.rfind(sub, start=1).fillna(-1),
        gs.str.rfind(sub, start=1),
        check_dtype=False,
    )
    assert_eq(
        ps.str.rfind(sub, end=10).fillna(-1),
        gs.str.rfind(sub, end=10),
        check_dtype=False,
    )
    assert_eq(
        ps.str.rfind(sub, start=2, end=10).fillna(-1),
        gs.str.rfind(sub, start=2, end=10),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "data,sub,er",
    [
        (["abc", "xyz", "a", "ab", "123", "097"], "a", ValueError),
        (["A B", "1.5", "3,000"], "abc", ValueError),
        (["23", "³", "⅕", ""], "⅕", ValueError),
        ([" ", "\t\r\n ", ""], "\n", ValueError),
        (["$", "B", "Aab$", "$$ca", "C$B$", "cat"], "$", ValueError),
        (["line to be wrapped", "another line to be wrapped"], " ", None),
        (
            ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
            "+",
            ValueError,
        ),
        (["line to be wrapped", "another line to be wrapped"], "", None),
    ],
)
def test_string_str_index(data, sub, er):
    ps = pd.Series(data)
    gs = Series(data)

    if er is None:
        assert_eq(ps.str.index(sub), gs.str.index(sub), check_dtype=False)

    try:
        ps.str.index(sub)
    except er:
        pass
    else:
        assert not er

    try:
        gs.str.index(sub)
    except er:
        pass
    else:
        assert not er


@pytest.mark.parametrize(
    "data,sub,er",
    [
        (["abc", "xyz", "a", "ab", "123", "097"], "a", ValueError),
        (["A B", "1.5", "3,000"], "abc", ValueError),
        (["23", "³", "⅕", ""], "⅕", ValueError),
        ([" ", "\t\r\n ", ""], "\n", ValueError),
        (["$", "B", "Aab$", "$$ca", "C$B$", "cat"], "$", ValueError),
        (["line to be wrapped", "another line to be wrapped"], " ", None),
        (
            ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
            "+",
            ValueError,
        ),
        (["line to be wrapped", "another line to be wrapped"], "", None),
    ],
)
def test_string_str_rindex(data, sub, er):
    ps = pd.Series(data)
    gs = Series(data)

    if er is None:
        assert_eq(ps.str.rindex(sub), gs.str.rindex(sub), check_dtype=False)

    try:
        ps.str.rindex(sub)
    except er:
        pass
    else:
        assert not er

    try:
        gs.str.rindex(sub)
    except er:
        pass
    else:
        assert not er


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
        ["line to be wrapped", "another line to be wrapped"],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize("pat", ["", " ", "a", "abc", "cat", "$", "\n"])
def test_string_str_match(data, pat):
    ps = pd.Series(data)
    gs = Series(data)

    assert_eq(ps.str.match(pat), gs.str.match(pat))


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
        ["line to be wrapped", "another line to be wrapped"],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
def test_string_str_translate(data):
    ps = pd.Series(data)
    gs = Series(data)

    assert_eq(
        ps.str.translate(str.maketrans({"a": "z"})),
        gs.str.translate(str.maketrans({"a": "z"})),
    )
    assert_eq(
        ps.str.translate(str.maketrans({"a": "z", "i": "$", "z": "1"})),
        gs.str.translate(str.maketrans({"a": "z", "i": "$", "z": "1"})),
    )
    assert_eq(
        ps.str.translate(
            str.maketrans({"+": "-", "-": "$", "?": "!", "B": "."})
        ),
        gs.str.translate(
            str.maketrans({"+": "-", "-": "$", "?": "!", "B": "."})
        ),
    )


def test_string_str_code_points():

    data = [
        "abc",
        "Def",
        None,
        "jLl",
        "dog and cat",
        "accénted",
        "",
        " 1234 ",
        "XYZ",
    ]
    gs = Series(data)
    expected = [
        97,
        98,
        99,
        68,
        101,
        102,
        106,
        76,
        108,
        100,
        111,
        103,
        32,
        97,
        110,
        100,
        32,
        99,
        97,
        116,
        97,
        99,
        99,
        50089,
        110,
        116,
        101,
        100,
        32,
        49,
        50,
        51,
        52,
        32,
        88,
        89,
        90,
    ]
    expected = Series(expected)

    assert_eq(expected, gs.str.code_points(), check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        ["http://www.hellow.com", "/home/nvidia/nfs", "123.45 ~ABCDEF"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
    ],
)
def test_string_str_url_encode(data):
    import urllib.parse

    gs = Series(data)

    got = gs.str.url_encode()
    expected = pd.Series([urllib.parse.quote(url, safe="~") for url in data])
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [
            "http://www.hellow.com?k1=acc%C3%A9nted&k2=a%2F/b.c",
            "%2Fhome%2fnfs",
            "987%20ZYX",
        ]
    ],
)
def test_string_str_decode_url(data):
    import urllib.parse

    gs = Series(data)

    got = gs.str.url_decode()
    expected = pd.Series([urllib.parse.unquote(url) for url in data])
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data,dtype",
    [
        (["0.1", "10.2", "10.876"], "float"),
        (["-0.1", "10.2", "+10.876"], "float"),
        (["1", "10.2", "10.876"], "float32"),
        (["+123", "6344556789", "0"], "int"),
        (["+123", "6344556789", "0"], "float"),
        (["0.1", "-10.2", "10.876", None], "float"),
    ],
)
@pytest.mark.parametrize("obj_type", [None, "str", "category"])
def test_string_typecast(data, obj_type, dtype):
    psr = pd.Series(data, dtype=obj_type)
    gsr = Series(data, dtype=obj_type)

    expect = psr.astype(dtype=dtype)
    actual = gsr.astype(dtype=dtype)
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "data,dtype",
    [
        (["0.1", "10.2", "10.876"], "int"),
        (["1", "10.2", "+10.876"], "int"),
        (["abc", "1", "2", " "], "int"),
        ([" ", "0.1", "2"], "float"),
        ([""], "int"),
        ([" "], "float"),
        (["\n"], "int"),
        (["0.1", "-10.2", "10.876", None], "int"),
        (["0.1", "-10.2", "10.876", None, "ab"], "float"),
        (["+", "-"], "float"),
        (["+", "-"], "int"),
        (["1++++", "--2"], "float"),
        (["1++++", "--2"], "int"),
        (["++++1", "--2"], "float"),
        (["++++1", "--2"], "int"),
    ],
)
@pytest.mark.parametrize("obj_type", [None, "str", "category"])
def test_string_typecast_error(data, obj_type, dtype):
    psr = pd.Series(data, dtype=obj_type)
    gsr = Series(data, dtype=obj_type)

    exception_type = None
    try:
        psr.astype(dtype=dtype)
    except Exception as e:
        exception_type = type(e)

    if exception_type is None:
        raise TypeError("Was expecting `psr.astype` to fail")

    with pytest.raises(exception_type):
        gsr.astype(dtype=dtype)


@pytest.mark.parametrize(
    "data",
    [
        ["f0:18:98:22:c2:e4", "00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff"],
        ["f0189822c2e4", "000000000000", "ffffffffffff"],
        ["0xf0189822c2e4", "0x000000000000", "0xffffffffffff"],
        ["0Xf0189822c2e4", "0X000000000000", "0Xffffffffffff"],
    ],
)
def test_string_hex_to_int(data):

    gsr = Series(data)

    got = gsr.str.htoi()
    expected = Series([263988422296292, 0, 281474976710655])

    assert_eq(expected, got)


def test_string_ip4_to_int():
    gsr = Series(["", None, "hello", "41.168.0.1", "127.0.0.1", "41.197.0.1"])
    expected = Series([0, None, 0, 698875905, 2130706433, 700776449])

    got = gsr.str.ip2int()

    assert_eq(expected, got)


def test_string_int_to_ipv4():
    gsr = Series([0, None, 0, 698875905, 2130706433, 700776449])
    expected = Series(
        ["0.0.0.0", None, "0.0.0.0", "41.168.0.1", "127.0.0.1", "41.197.0.1"]
    )

    got = Series(gsr._column.int2ip())

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("int8"),
        np.dtype("int16"),
        np.dtype("int32"),
        np.dtype("float32"),
        np.dtype("float64"),
    ],
)
def test_string_int_to_ipv4_dtype_fail(dtype):
    gsr = Series([1, 2, 3, 4, 5]).astype(dtype)
    with pytest.raises(TypeError):
        gsr._column.int2ip()
