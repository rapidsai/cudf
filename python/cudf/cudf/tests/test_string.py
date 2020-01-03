# Copyright (c) 2018, NVIDIA CORPORATION.

from contextlib import ExitStack as does_not_raise

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numba import cuda

import rmm

from cudf import concat
from cudf.core import DataFrame, Series
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
        rmm.to_device(np.array([0, 1, 2, 3, 4])),
    ],
)
def test_string_get_item(ps_gs, item):
    ps, gs = ps_gs

    got = gs[item]
    if isinstance(got, Series):
        got = got.to_arrow()

    if isinstance(item, cuda.devicearray.DeviceNDArray):
        item = item.copy_to_host()

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
        rmm.to_device(np.array([True] * 5)),
        rmm.to_device(np.array([False] * 5)),
        np.random.randint(0, 2, 5).astype("bool").tolist(),
        np.random.randint(0, 2, 5).astype("bool"),
        rmm.to_device(np.random.randint(0, 2, 5).astype("bool")),
    ],
)
def test_string_bool_mask(ps_gs, item):
    ps, gs = ps_gs

    got = gs[item]
    if isinstance(got, Series):
        got = got.to_arrow()

    if isinstance(item, cuda.devicearray.DeviceNDArray):
        item = item.copy_to_host()

    expect = ps[item]
    if isinstance(expect, pd.Series):
        expect = pa.Array.from_pandas(expect)
        pa.Array.equals(expect, got)
    else:
        assert expect == got


@pytest.mark.parametrize("item", [0, slice(1, 3), slice(5)])
def test_string_repr(ps_gs, item):
    ps, gs = ps_gs

    got_out = gs[item]
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
    [("a", False), ("f", False), (r"[a-z]", True), (r"[A-Z]", True)],
)
@pytest.mark.parametrize("case,case_raise", [(True, 0), (False, 1)])
@pytest.mark.parametrize("flags,flags_raise", [(0, 0), (1, 1)])
@pytest.mark.parametrize("na,na_raise", [(np.nan, 0), (None, 1), ("", 1)])
def test_string_contains(
    ps_gs, pat, regex, case, case_raise, flags, flags_raise, na, na_raise
):
    ps, gs = ps_gs

    expectation = raise_builder(
        [case_raise, flags_raise, na_raise], NotImplementedError
    )

    with expectation:
        expect = ps.str.contains(
            pat, case=case, flags=flags, na=na, regex=regex
        )
        got = gs.str.contains(pat, case=case, flags=flags, na=na, regex=regex)

        expect = pa.array(expect, from_pandas=True).cast(pa.bool_())
        got = got.to_arrow()

        assert pa.Array.equals(expect, got)


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
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = as_index(as_column(["a", "b", "c", "d", "e"]), name="name")
    pdf.index = stringIndex
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
