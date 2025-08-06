# Copyright (c) 2018-2025, NVIDIA CORPORATION.

from decimal import Decimal
from sys import getsizeof

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import rmm

import cudf
from cudf.core.buffer import as_buffer
from cudf.core.column.string import StringColumn
from cudf.core.index import Index
from cudf.testing import assert_eq
from cudf.testing._utils import (
    DATETIME_TYPES,
    NUMERIC_TYPES,
    assert_exceptions_equal,
)
from cudf.utils import dtypes as dtypeutils


@pytest.fixture(
    params=[
        ["AbC", "de", "FGHI", "j", "kLm"],
        ["nOPq", None, "RsT", None, "uVw"],
        [None, None, None, None, None],
    ],
    ids=["no_nulls", "some_nulls", "all_nulls"],
)
def data(request):
    return request.param


@pytest.fixture(
    params=[None, [10, 11, 12, 13, 14]], ids=["None_index", "Set_index"]
)
def index(request):
    return request.param


@pytest.fixture
def ps_gs(data, index):
    ps = pd.Series(data, index=index, dtype="str", name="nice name")
    gs = cudf.Series(data, index=index, dtype="str", name="nice name")
    return (ps, gs)


@pytest.mark.parametrize("construct", [list, np.array, pd.Series, pa.array])
def test_string_ingest(construct):
    expect = ["a", "a", "b", "c", "a"]
    data = construct(expect)
    got = cudf.Series(data)
    assert got.dtype == np.dtype("object")
    assert len(got) == 5
    for idx, val in enumerate(expect):
        assert expect[idx] == got[idx]


def test_string_export(ps_gs):
    ps, gs = ps_gs

    expect = ps
    got = gs.to_pandas()
    assert_eq(expect, got)

    expect = np.array(ps)
    got = gs.to_numpy()
    assert_eq(expect, got)

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
    if isinstance(got, cudf.Series):
        got = got.to_arrow()

    if isinstance(item, cupy.ndarray):
        item = cupy.asnumpy(item)

    expect = ps.iloc[item]
    if isinstance(expect, pd.Series):
        expect = pa.Array.from_pandas(expect)
        pa.Array.equals(expect, got)
    else:
        if got is cudf.NA and expect is None:
            return
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
        np.random.default_rng(seed=0)
        .integers(0, 2, 5)
        .astype("bool")
        .tolist(),
        np.random.default_rng(seed=0).integers(0, 2, 5).astype("bool"),
        cupy.asarray(
            np.random.default_rng(seed=0).integers(0, 2, 5).astype("bool")
        ),
    ],
)
def test_string_bool_mask(ps_gs, item):
    ps, gs = ps_gs

    got = gs.iloc[item]
    if isinstance(got, cudf.Series):
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

    if got_out is not cudf.NA and len(got_out) > 1:
        expect = expect.replace("None", "<NA>")

    assert expect == got or (expect == "None" and got == "<NA>")


@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + ["bool", "object", "str"]
)
def test_string_astype(dtype):
    if (
        dtype.startswith("int")
        or dtype.startswith("uint")
        or dtype.startswith("long")
    ):
        data = ["1", "2", "3", "4", "5"]
    elif dtype.startswith("float"):
        data = [
            "1.0",
            "2.0",
            "3.0",
            "4.0",
            None,
            "5.0",
            "nan",
            "-INF",
            "NaN",
            "inF",
            "NAn",
        ]
    elif dtype.startswith("bool"):
        data = ["True", "False", "True", "False", "False"]
    elif dtype.startswith("datetime64"):
        data = [
            "2019-06-04T00:00:00",
            "2019-06-04T12:12:12",
            "2019-06-03T00:00:00",
            "2019-05-04T00:00:00",
            "2018-06-04T00:00:00",
            "1922-07-21T01:02:03",
        ]
    elif dtype == "str" or dtype == "object":
        data = ["ab", "cd", "ef", "gh", "ij"]
    ps = pd.Series(data)
    gs = cudf.Series(data)

    expect = ps.astype(dtype)
    got = gs.astype(dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data, scale, precision",
    [
        (["1.11", "2.22", "3.33"], 2, 3),
        (["111", "222", "33"], 0, 3),
        (["111000", "22000", "3000"], -3, 3),
        ([None, None, None], 0, 5),
        ([None, "-2345", None], 0, 5),
        ([], 0, 5),
    ],
)
@pytest.mark.parametrize(
    "decimal_dtype",
    [cudf.Decimal128Dtype, cudf.Decimal64Dtype, cudf.Decimal32Dtype],
)
def test_string_to_decimal(data, scale, precision, decimal_dtype):
    gs = cudf.Series(data, dtype="str")
    fp = gs.astype(decimal_dtype(scale=scale, precision=precision))
    got = fp.astype("str")
    assert_eq(gs, got)


def test_string_empty_to_decimal():
    gs = cudf.Series(["", "-85", ""], dtype="str")
    got = gs.astype(cudf.Decimal64Dtype(scale=0, precision=5))
    expected = cudf.Series(
        [0, -85, 0],
        dtype=cudf.Decimal64Dtype(scale=0, precision=5),
    )
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data, scale, precision",
    [
        (["1.23", "-2.34", "3.45"], 2, 3),
        (["123", "-234", "345"], 0, 3),
        (["12300", "-400", "5000.0"], -2, 5),
        ([None, None, None], 0, 5),
        ([None, "-100", None], 0, 5),
        ([], 0, 5),
    ],
)
@pytest.mark.parametrize(
    "decimal_dtype",
    [cudf.Decimal128Dtype, cudf.Decimal32Dtype, cudf.Decimal64Dtype],
)
def test_string_from_decimal(data, scale, precision, decimal_dtype):
    decimal_data = []
    for d in data:
        if d is None:
            decimal_data.append(None)
        else:
            decimal_data.append(Decimal(d))
    fp = cudf.Series(
        decimal_data,
        dtype=decimal_dtype(scale=scale, precision=precision),
    )
    gs = fp.astype("str")
    got = gs.astype(decimal_dtype(scale=scale, precision=precision))
    assert_eq(fp, got)


@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + ["bool", "object", "str"]
)
def test_string_empty_astype(dtype):
    data = []
    ps = pd.Series(data, dtype="str")
    gs = cudf.Series(data, dtype="str")

    expect = ps.astype(dtype)
    got = gs.astype(dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES + DATETIME_TYPES + ["bool"])
def test_string_numeric_astype(dtype):
    if dtype.startswith("bool"):
        data = [1, 0, 1, 0, 1]
    elif (
        dtype.startswith("int")
        or dtype.startswith("uint")
        or dtype.startswith("long")
    ):
        data = [1, 2, 3, 4, 5]
    elif dtype.startswith("float"):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
    elif dtype.startswith("datetime64"):
        # pandas rounds the output format based on the data
        # Use numpy instead
        # but fix '2011-01-01T00:00:00' -> '2011-01-01 00:00:00'
        data = [1000000001, 2000000001, 3000000001, 4000000001, 5000000001]
        ps = np.asarray(data, dtype=dtype).astype(str)
        ps = np.array([i.replace("T", " ") for i in ps])

    if not dtype.startswith("datetime64"):
        ps = pd.Series(data, dtype=dtype)

    gs = cudf.Series(data, dtype=dtype)

    expect = pd.Series(ps.astype("str"))
    got = gs.astype("str")

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES + DATETIME_TYPES + ["bool"])
def test_string_empty_numeric_astype(dtype):
    data = []

    if dtype.startswith("datetime64"):
        ps = pd.Series(data, dtype="datetime64[ns]")
    else:
        ps = pd.Series(data, dtype=dtype)
    gs = cudf.Series(data, dtype=dtype)

    expect = ps.astype("str")
    got = gs.astype("str")

    assert_eq(expect, got)


@pytest.mark.parametrize("ascending", [True, False])
def test_string_sort(ps_gs, ascending):
    ps, gs = ps_gs

    expect = ps.sort_values(ascending=ascending)
    got = gs.sort_values(ascending=ascending)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "str_data", [[], ["a", "b", "c", "d", "e"], [None, None, None, None, None]]
)
@pytest.mark.parametrize("num_keys", [1, 2, 3])
def test_string_groupby_key(str_data, num_keys):
    other_data = [1, 2, 3, 4, 5][: len(str_data)]

    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()
    for i in range(num_keys):
        pdf[i] = pd.Series(str_data, dtype="str")
        gdf[i] = cudf.Series(str_data, dtype="str")
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
    gdf = cudf.DataFrame()
    for i in range(num_cols):
        pdf[i] = pd.Series(str_data, dtype="str")
        gdf[i] = cudf.Series(str_data, dtype="str")
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
    gdf = cudf.DataFrame()
    pdf["a"] = pd.Series(str_data, dtype="str")
    gdf["a"] = cudf.Series(str_data, dtype="str")
    pdf["b"] = other_data
    gdf["b"] = other_data

    expect = pdf.groupby("a", sort=True).count()
    got = gdf.groupby("a", sort=True).count()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("scalar", ["a", None])
def test_string_set_scalar(scalar):
    pdf = pd.DataFrame()
    pdf["a"] = [1, 2, 3, 4, 5]
    gdf = cudf.DataFrame.from_pandas(pdf)

    pdf["b"] = "a"
    gdf["b"] = "a"

    assert_eq(pdf["b"], gdf["b"])
    assert_eq(pdf, gdf)


def test_string_index():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(5, 5)))
    gdf = cudf.DataFrame.from_pandas(pdf)
    stringIndex = ["a", "b", "c", "d", "e"]
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = np.array(["a", "b", "c", "d", "e"])
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = Index(["a", "b", "c", "d", "e"], name="name")
    pdf.index = stringIndex.to_pandas()
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = cudf.Index._from_column(
        cudf.core.column.as_column(["a", "b", "c", "d", "e"]), name="name"
    )
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
    gs = cudf.Series(item)
    # Pandas `unique` returns a numpy array
    pres = pd.Series(ps.unique())
    # cudf returns a cudf.Series
    gres = gs.unique()
    assert_eq(pres, gres)


def test_string_equality():
    data1 = ["b", "c", "d", "a", "c"]
    data2 = ["a", None, "c", "a", "c"]

    ps1 = pd.Series(data1)
    ps2 = pd.Series(data2)
    gs1 = cudf.Series(data1)
    gs2 = cudf.Series(data2)

    expect = ps1 == ps2
    got = gs1 == gs2

    assert_eq(expect, got.fillna(False))

    expect = ps1 == "m"
    got = gs1 == "m"

    assert_eq(expect, got.fillna(False))

    ps1 = pd.Series(["a"])
    gs1 = cudf.Series(["a"])

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
    gds = cudf.Series(lhs) + cudf.Series(rhs)

    assert_eq(pds, gds)


def test_string_no_children_properties():
    empty_col = StringColumn(
        as_buffer(rmm.DeviceBuffer(size=0)),
        size=0,
        dtype=np.dtype("object"),
        children=(),
    )
    assert empty_col.base_children == ()
    assert empty_col.base_size == 0

    assert empty_col.children == ()
    assert empty_col.size == 0

    assert getsizeof(empty_col) >= 0  # Accounts for Python GC overhead


def test_string_table_view_creation():
    data = ["hi"] * 25 + [None] * 2027
    psr = pd.Series(data)
    gsr = cudf.Series.from_pandas(psr)

    expect = psr[:1]
    got = gsr[:1]

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data,dtype",
    [
        (["0.1", "10.2", "10.876"], "float"),
        (["-0.1", "10.2", "+10.876"], "float"),
        (["1", "10.2", "10.876"], "float32"),
        (["+123", "6344556789", "0"], "int"),
        (["+123", "6344556789", "0"], "uint64"),
        (["+123", "6344556789", "0"], "float"),
        (["0.1", "-10.2", "10.876", None], "float"),
    ],
)
@pytest.mark.parametrize("obj_type", [None, "str", "category"])
def test_string_typecast(data, obj_type, dtype):
    psr = pd.Series(data, dtype=obj_type)
    gsr = cudf.Series(data, dtype=obj_type)

    expect = psr.astype(dtype=dtype)
    actual = gsr.astype(dtype=dtype)
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "data,dtype",
    [
        (["0.1", "10.2", "10.876"], "int"),
        (["1", "10.2", "+10.876"], "int"),
        (["abc", "1", "2", " "], "int"),
        (["0.1", "10.2", "10.876"], "uint64"),
        (["1", "10.2", "+10.876"], "uint64"),
        (["abc", "1", "2", " "], "uint64"),
        ([" ", "0.1", "2"], "float"),
        ([""], "int"),
        ([""], "uint64"),
        ([" "], "float"),
        (["\n"], "int"),
        (["\n"], "uint64"),
        (["0.1", "-10.2", "10.876", None], "int"),
        (["0.1", "-10.2", "10.876", None], "uint64"),
        (["0.1", "-10.2", "10.876", None, "ab"], "float"),
        (["+", "-"], "float"),
        (["+", "-"], "int"),
        (["+", "-"], "uint64"),
        (["1++++", "--2"], "float"),
        (["1++++", "--2"], "int"),
        (["1++++", "--2"], "uint64"),
        (["++++1", "--2"], "float"),
        (["++++1", "--2"], "int"),
        (["++++1", "--2"], "uint64"),
    ],
)
@pytest.mark.parametrize("obj_type", [None, "str", "category"])
def test_string_typecast_error(data, obj_type, dtype):
    psr = pd.Series(data, dtype=obj_type)
    gsr = cudf.Series(data, dtype=obj_type)

    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=([dtype],),
        rfunc_args_and_kwargs=([dtype],),
    )


def test_string_int_to_ipv4():
    gsr = cudf.Series([0, None, 0, 698875905, 2130706433, 700776449]).astype(
        "uint32"
    )
    expected = cudf.Series(
        ["0.0.0.0", None, "0.0.0.0", "41.168.0.1", "127.0.0.1", "41.197.0.1"]
    )

    got = cudf.Series._from_column(gsr._column.int2ip())

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "dtype", sorted(list(dtypeutils.NUMERIC_TYPES - {"uint32"}))
)
def test_string_int_to_ipv4_dtype_fail(dtype):
    gsr = cudf.Series([1, 2, 3, 4, 5]).astype(dtype)
    with pytest.raises(TypeError):
        gsr._column.int2ip()


@pytest.mark.parametrize(
    "data",
    [
        ["a", "b", "c", "d", "e"],
        ["a", "z", ".", '"', "aa", "zz"],
        ["aa", "zz"],
        ["z", "a", "zz", "aa"],
        ["1", "2", "3", "4", "5"],
        [""],
        ["a"],
        ["hello"],
        ["small text", "this is a larger text......"],
        ["👋🏻", "🔥", "🥇"],
        ["This is 💯", "here is a calendar", "📅"],
        ["", ".", ";", "[", "]"],
        ["\t", ".", "\n", "\n\t", "\t\n"],
    ],
)
def test_str_min(data):
    psr = pd.Series(data)
    sr = cudf.Series(data)

    assert_eq(psr.min(), sr.min())


@pytest.mark.parametrize(
    "data",
    [
        ["a", "b", "c", "d", "e"],
        ["a", "z", ".", '"', "aa", "zz"],
        ["aa", "zz"],
        ["z", "a", "zz", "aa"],
        ["1", "2", "3", "4", "5"],
        [""],
        ["a"],
        ["hello"],
        ["small text", "this is a larger text......"],
        ["👋🏻", "🔥", "🥇"],
        ["This is 💯", "here is a calendar", "📅"],
        ["", ".", ";", "[", "]"],
        ["\t", ".", "\n", "\n\t", "\t\n"],
    ],
)
def test_str_max(data):
    psr = pd.Series(data)
    sr = cudf.Series(data)

    assert_eq(psr.max(), sr.max())


@pytest.mark.parametrize(
    "data",
    [
        ["a", "b", "c", "d", "e"],
        ["a", "z", ".", '"', "aa", "zz"],
        ["aa", "zz"],
        ["z", "a", "zz", "aa"],
        ["1", "2", "3", "4", "5"],
        [""],
        ["a"],
        ["hello"],
        ["small text", "this is a larger text......"],
        ["👋🏻", "🔥", "🥇"],
        ["This is 💯", "here is a calendar", "📅"],
        ["", ".", ";", "[", "]"],
        ["\t", ".", "\n", "\n\t", "\t\n"],
    ],
)
def test_str_sum(data):
    psr = pd.Series(data)
    sr = cudf.Series(data)

    assert_eq(psr.sum(), sr.sum())


def test_str_mean():
    sr = cudf.Series(["a", "b", "c", "d", "e"])

    with pytest.raises(TypeError):
        sr.mean()


def test_string_product():
    psr = pd.Series(["1", "2", "3", "4", "5"])
    sr = cudf.Series(["1", "2", "3", "4", "5"])

    assert_exceptions_equal(
        lfunc=psr.product,
        rfunc=sr.product,
    )


def test_string_var():
    psr = pd.Series(["1", "2", "3", "4", "5"])
    sr = cudf.Series(["1", "2", "3", "4", "5"])

    assert_exceptions_equal(lfunc=psr.var, rfunc=sr.var)


def test_string_std():
    psr = pd.Series(["1", "2", "3", "4", "5"])
    sr = cudf.Series(["1", "2", "3", "4", "5"])

    assert_exceptions_equal(lfunc=psr.std, rfunc=sr.std)


def test_string_slice_with_mask():
    actual = cudf.Series(["hi", "hello", None])
    expected = actual[0:3]

    assert actual._column.base_size == 3
    assert_eq(actual._column.base_size, expected._column.base_size)
    assert_eq(actual._column.null_count, expected._column.null_count)

    assert_eq(actual, expected)


def test_string_reduction_error():
    s = cudf.Series([None, None], dtype="str")
    ps = s.to_pandas(nullable=True)
    assert_exceptions_equal(
        s.any,
        ps.any,
        lfunc_args_and_kwargs=([], {"skipna": False}),
        rfunc_args_and_kwargs=([], {"skipna": False}),
    )

    assert_exceptions_equal(
        s.all,
        ps.all,
        lfunc_args_and_kwargs=([], {"skipna": False}),
        rfunc_args_and_kwargs=([], {"skipna": False}),
    )
