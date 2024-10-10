# Copyright (c) 2018-2024, NVIDIA CORPORATION.


import datetime
import inspect
from itertools import product

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import DataFrame
from cudf.testing import assert_eq
from cudf.utils import queryutils

_params_query_parser = []
_params_query_parser.append(("a > @b", ("a", "__CUDF_ENVREF__b")))
_params_query_parser.append(("(a + b) <= @c", ("a", "b", "__CUDF_ENVREF__c")))
_params_query_parser.append(("a > b if a > 0 else b > a", ("a", "b")))


@pytest.mark.parametrize("text,expect_args", _params_query_parser)
def test_query_parser(text, expect_args):
    info = queryutils.query_parser(text)
    fn = queryutils.query_builder(info, "myfoo")
    assert callable(fn)
    argspec = inspect.getfullargspec(fn)
    assert tuple(argspec.args) == tuple(expect_args)


params_query_data = list(product([1, 2, 7, 8, 9, 16, 100, 129], range(2)))
params_query_fn = [
    (lambda a, b: a < b, "a < b"),
    (lambda a, b: a * 2 >= b, "a * 2 >= b"),
    (lambda a, b: 2 * (a + b) > (a + b) / 2, "2 * (a + b) > (a + b) / 2"),
]
nulls = [True, False]


@pytest.mark.parametrize(
    "data,fn,nulls", product(params_query_data, params_query_fn, nulls)
)
def test_query(data, fn, nulls):
    # prepare
    nelem, seed = data
    expect_fn, query_expr = fn
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame()
    pdf["a"] = np.arange(nelem)
    pdf["b"] = rng.random(nelem) * nelem
    if nulls:
        pdf.loc[::2, "a"] = None
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.query(query_expr), gdf.query(query_expr))


params_query_env_fn = [
    (lambda a, b, c, d: a * c > b + d, "a * @c > b + @d"),
    (
        lambda a, b, c, d: ((a / c) < d) | ((b**c) > d),
        "((a / @c) < @d) | ((b ** @c) > @d)",
    ),
]


@pytest.mark.parametrize(
    "data,fn", product(params_query_data, params_query_env_fn)
)
def test_query_ref_env(data, fn):
    # prepare
    nelem, seed = data
    expect_fn, query_expr = fn
    rng = np.random.default_rng(seed=0)
    df = DataFrame()
    df["a"] = aa = np.arange(nelem)
    df["b"] = bb = rng.random(nelem) * nelem
    c = 2.3
    d = 1.2
    # udt
    expect_mask = expect_fn(aa, bb, c, d)
    print(expect_mask)
    df2 = df.query(query_expr)
    # check
    assert len(df2) == np.count_nonzero(expect_mask)
    np.testing.assert_array_almost_equal(df2["a"].to_numpy(), aa[expect_mask])
    np.testing.assert_array_almost_equal(df2["b"].to_numpy(), bb[expect_mask])


def test_query_env_changing():
    df = DataFrame()
    df["a"] = aa = np.arange(100)
    expr = "a < @c"
    # first attempt
    c = 10
    got = df.query(expr)
    np.testing.assert_array_equal(aa[aa < c], got["a"].to_numpy())
    # change env
    c = 50
    got = df.query(expr)
    np.testing.assert_array_equal(aa[aa < c], got["a"].to_numpy())


def test_query_local_dict():
    df = DataFrame()
    df["a"] = aa = np.arange(100)
    expr = "a < @val"

    got = df.query(expr, local_dict={"val": 10})
    np.testing.assert_array_equal(aa[aa < 10], got["a"].to_numpy())

    # test for datetime
    df = DataFrame()
    data = np.array(["2018-10-07", "2018-10-08"], dtype="datetime64")
    df["datetimes"] = data
    search_date = datetime.datetime.strptime("2018-10-08", "%Y-%m-%d")
    expr = "datetimes==@search_date"

    got = df.query(expr, local_dict={"search_date": search_date})
    np.testing.assert_array_equal(data[1], got["datetimes"].to_numpy())


def test_query_splitted_combine():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {"x": rng.integers(0, 5, size=10), "y": rng.normal(size=10)}
    )
    gdf = DataFrame.from_pandas(df)

    # Split the GDF
    s1 = gdf[:5]
    s2 = gdf[5:]

    # Do the query
    expr = "x > 2"
    q1 = s1.query(expr)
    q2 = s2.query(expr)
    # Combine
    got = cudf.concat([q1, q2]).to_pandas()

    # Should equal to just querying the original GDF
    expect = gdf.query(expr).to_pandas()
    assert_eq(got, expect, check_index_type=True)


def test_query_empty_frames():
    empty_pdf = pd.DataFrame({"a": [], "b": []})
    empty_gdf = DataFrame.from_pandas(empty_pdf)
    # Do the query
    expr = "a > 2"
    got = empty_gdf.query(expr).to_pandas()
    expect = empty_pdf.query(expr)

    # assert equal results
    assert_eq(got, expect)


@pytest.mark.parametrize(("a_val", "b_val", "c_val"), [(4, 3, 15)])
@pytest.mark.parametrize("index", ["a", ["a", "b"]])
@pytest.mark.parametrize(
    "query",
    [
        "a < @a_val",
        "a < @a_val and b > @b_val",
        "(a < @a_val and b >@b_val) or c >@c_val",
    ],
)
def test_query_with_index_name(index, query, a_val, b_val, c_val):
    pdf = pd.DataFrame(
        {
            "a": [1, None, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [12, 15, 17, 19, 27],
        }
    )
    pdf.set_index(index)

    gdf = DataFrame.from_pandas(pdf)

    out = gdf.query(query)
    expect = pdf.query(query)

    assert_eq(out, expect)


@pytest.mark.parametrize(("a_val", "b_val", "c_val"), [(4, 3, 15)])
@pytest.mark.parametrize(
    "query",
    [
        "index < @a_val",
        "index < @a_val and b > @b_val",
        "(index < @a_val and b >@b_val) or c >@c_val",
    ],
)
def test_query_with_index_keyword(query, a_val, b_val, c_val):
    pdf = pd.DataFrame(
        {
            "a": [1, None, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [12, 15, 17, 19, 27],
        }
    )
    pdf.set_index("a")

    gdf = DataFrame.from_pandas(pdf)

    out = gdf.query(query)
    expect = pdf.query(query)

    assert_eq(out, expect)


@pytest.mark.parametrize(
    "data, query",
    [
        # Only need to test the dtypes that pandas
        # supports but that we do not
        (["a", "b", "c"], "data == 'a'"),
    ],
)
def test_query_unsupported_dtypes(data, query):
    gdf = cudf.DataFrame({"data": data})

    # make sure the query works in pandas
    pdf = gdf.to_pandas()
    pdf_result = pdf.query(query)

    expect = pd.DataFrame({"data": ["a"]})
    assert_eq(expect, pdf_result)

    # but fails in cuDF
    with pytest.raises(TypeError):
        gdf.query(query)
