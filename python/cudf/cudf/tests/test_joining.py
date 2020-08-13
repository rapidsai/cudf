# Copyright (c) 2018, NVIDIA CORPORATION.
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import DataFrame, Series
from cudf.tests.utils import INTEGER_TYPES, NUMERIC_TYPES, assert_eq


def make_params():
    np.random.seed(0)

    hows = "left,inner,outer,right,leftanti,leftsemi".split(",")
    methods = "hash,sort".split(",")

    # Test specific cases (1)
    aa = [0, 0, 4, 5, 5]
    bb = [0, 0, 2, 3, 5]
    for how in hows:
        if how in ["left", "inner", "right", "leftanti", "leftsemi"]:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield (aa, bb, how, "sort")

    # Test specific cases (2)
    aa = [0, 0, 1, 2, 3]
    bb = [0, 1, 2, 2, 3]
    for how in hows:
        if how in ["left", "inner", "right", "leftanti", "leftsemi"]:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield (aa, bb, how, "sort")

    # Test large random integer inputs
    aa = np.random.randint(0, 50, 100)
    bb = np.random.randint(0, 50, 100)
    for how in hows:
        if how in ["left", "inner", "right", "leftanti", "leftsemi"]:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield (aa, bb, how, "sort")

    # Test floating point inputs
    aa = np.random.random(50)
    bb = np.random.random(50)
    for how in hows:
        if how in ["left", "inner", "right", "leftanti", "leftsemi"]:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield (aa, bb, how, "sort")


def pd_odd_joins(left, right, join_type):
    if join_type == "leftanti":
        return left[~left.index.isin(right.index)][left.columns]
    elif join_type == "leftsemi":
        return left[left.index.isin(right.index)][left.columns]


@pytest.mark.parametrize("aa,bb,how,method", make_params())
def test_dataframe_join_how(aa, bb, how, method):
    df = DataFrame()
    df["a"] = aa
    df["b"] = bb

    def work_pandas(df, how):
        df1 = df.set_index("a")
        df2 = df.set_index("b")
        if how == "leftanti":
            joined = pd_odd_joins(df1, df2, "leftanti")
        elif how == "leftsemi":
            joined = pd_odd_joins(df1, df2, "leftsemi")
        else:
            joined = df1.join(df2, how=how, sort=True)
        return joined

    def work_gdf(df):
        df1 = df.set_index("a")
        df2 = df.set_index("b")
        joined = df1.join(df2, how=how, sort=True, method=method)
        return joined

    expect = work_pandas(df.to_pandas(), how)
    got = work_gdf(df)
    expecto = expect.copy()
    goto = got.copy()

    expect = expect.astype(np.float64).fillna(np.nan)[expect.columns]
    got = got.astype(np.float64).fillna(np.nan)[expect.columns]

    assert got.index.name is None

    assert list(expect.columns) == list(got.columns)
    # test disabled until libgdf sort join gets updated with new api
    if method == "hash":
        assert_eq(sorted(expect.index.values), sorted(got.index.values))
        if how != "outer":
            # Newly introduced ambiguous ValueError thrown when
            # an index and column have the same name. Rename the
            # index so sorts work.
            # TODO: What is the less hacky way?
            expect.index.name = "bob"
            got.index.name = "mary"
            assert_eq(
                got.sort_values(got.columns.to_list()).reset_index(drop=True),
                expect.sort_values(expect.columns.to_list()).reset_index(
                    drop=True
                ),
            )
        # if(how=='right'):
        #     _sorted_check_series(expect['a'], expect['b'],
        #                          got['a'], got['b'])
        # else:
        #     _sorted_check_series(expect['b'], expect['a'], got['b'],
        #                          got['a'])
        else:
            for c in expecto.columns:
                _check_series(expecto[c].fillna(-1), goto[c].fillna(-1))


def _check_series(expect, got):
    magic = 0xDEADBEAF
    # print("expect\n", expect)
    # print("got\n", got.to_string(nrows=None))
    direct_equal = np.all(expect.values == got.to_array())
    nanfilled_equal = np.all(
        expect.fillna(magic).values == got.fillna(magic).to_array()
    )
    msg = "direct_equal={}, nanfilled_equal={}".format(
        direct_equal, nanfilled_equal
    )
    assert direct_equal or nanfilled_equal, msg


def test_dataframe_join_suffix():
    np.random.seed(0)

    df = DataFrame()
    for k in "abc":
        df[k] = np.random.randint(0, 5, 5)

    left = df.set_index("a")
    right = df.set_index("c")
    with pytest.raises(ValueError) as raises:
        left.join(right)
    raises.match(
        "there are overlapping columns but lsuffix"
        " and rsuffix are not defined"
    )

    got = left.join(right, lsuffix="_left", rsuffix="_right", sort=True)
    # Get expected value
    pddf = df.to_pandas()
    expect = pddf.set_index("a").join(
        pddf.set_index("c"), lsuffix="_left", rsuffix="_right"
    )
    # Check
    assert list(expect.columns) == list(got.columns)
    assert_eq(expect.index.values, got.index.values)
    for k in expect.columns:
        _check_series(expect[k].fillna(-1), got[k].fillna(-1))


def test_dataframe_join_cats():
    lhs = DataFrame()
    lhs["a"] = pd.Categorical(list("aababcabbc"), categories=list("abc"))
    lhs["b"] = bb = np.arange(len(lhs))
    lhs = lhs.set_index("a")

    rhs = DataFrame()
    rhs["a"] = pd.Categorical(list("abcac"), categories=list("abc"))
    rhs["c"] = cc = np.arange(len(rhs))
    rhs = rhs.set_index("a")

    got = lhs.join(rhs)
    expect = lhs.to_pandas(nullable_pd_dtype=False).join(
        rhs.to_pandas(nullable_pd_dtype=False)
    )

    # Note: pandas make an object Index after joining
    assert_eq(
        got.sort_values(by="b").sort_index().reset_index(drop=True),
        expect.reset_index(drop=True),
    )

    # Just do some rough checking here.
    assert list(got.columns) == ["b", "c"]
    assert len(got) > 0
    assert set(got.index.to_pandas()) & set("abc")
    assert set(got["b"].to_array()) & set(bb)
    assert set(got["c"].to_array()) & set(cc)


def test_dataframe_join_combine_cats():
    lhs = cudf.DataFrame({"join_index": ["a", "b", "c"], "data_x": [1, 2, 3]})
    rhs = cudf.DataFrame({"join_index": ["b", "c", "d"], "data_y": [2, 3, 4]})

    lhs["join_index"] = lhs["join_index"].astype("category")
    rhs["join_index"] = rhs["join_index"].astype("category")

    lhs = lhs.set_index("join_index")
    rhs = rhs.set_index("join_index")

    lhs_pd = lhs.to_pandas()
    rhs_pd = rhs.to_pandas()

    lhs_pd.index = lhs_pd.index.astype("object")
    rhs_pd.index = rhs_pd.index.astype("object")

    expect = lhs_pd.join(rhs_pd, how="outer")
    expect.index = expect.index.astype("category")
    got = lhs.join(rhs, how="outer")

    # TODO: Remove copying to host
    # after https://github.com/rapidsai/cudf/issues/5676
    # is implemented
    assert_eq(expect.index.sort_values(), got.index.to_pandas().sort_values())


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_dataframe_join_mismatch_cats(how):
    pdf1 = pd.DataFrame(
        {
            "join_col": ["a", "b", "c", "d", "e"],
            "data_col_left": [10, 20, 30, 40, 50],
        }
    )
    pdf2 = pd.DataFrame(
        {"join_col": ["c", "e", "f"], "data_col_right": [6, 7, 8]}
    )

    pdf1["join_col"] = pdf1["join_col"].astype("category")
    pdf2["join_col"] = pdf2["join_col"].astype("category")

    gdf1 = DataFrame.from_pandas(pdf1)
    gdf2 = DataFrame.from_pandas(pdf2)

    gdf1 = gdf1.set_index("join_col")
    gdf2 = gdf2.set_index("join_col")

    pdf1 = pdf1.set_index("join_col")
    pdf2 = pdf2.set_index("join_col")
    join_gdf = gdf1.join(gdf2, how=how, sort=True, method="hash")
    join_pdf = pdf1.join(pdf2, how=how)

    got = join_gdf.fillna(-1).to_pandas(nullable_pd_dtype=False)
    expect = join_pdf.fillna(-1)  # note: cudf join doesn't mask NA

    # We yield a categorical here whereas pandas gives Object.
    expect.index = expect.index.astype("category")
    # cudf creates the columns in different order than pandas for right join
    if how == "right":
        got = got[["data_col_left", "data_col_right"]]

    expect.data_col_right = expect.data_col_right.astype(np.int64)
    expect.data_col_left = expect.data_col_left.astype(np.int64)

    assert_eq(expect, got)


@pytest.mark.parametrize("on", ["key1", ["key1", "key2"], None])
def test_dataframe_merge_on(on):
    np.random.seed(0)

    # Make cuDF
    df_left = DataFrame()
    nelem = 500
    df_left["key1"] = np.random.randint(0, 40, nelem)
    df_left["key2"] = np.random.randint(0, 50, nelem)
    df_left["left_val"] = np.arange(nelem)

    df_right = DataFrame()
    nelem = 500
    df_right["key1"] = np.random.randint(0, 30, nelem)
    df_right["key2"] = np.random.randint(0, 50, nelem)
    df_right["right_val"] = np.arange(nelem)

    # Make pandas DF
    pddf_left = df_left.to_pandas(nullable_pd_dtype=False)
    pddf_right = df_right.to_pandas(nullable_pd_dtype=False)

    # Expected result (from pandas)
    pddf_joined = pddf_left.merge(pddf_right, on=on, how="left")

    # Test (from cuDF; doesn't check for ordering)
    join_result = df_left.merge(df_right, on=on, how="left")
    join_result_cudf = cudf.merge(df_left, df_right, on=on, how="left")

    join_result["right_val"] = (
        join_result["right_val"].astype(np.float64).fillna(np.nan)
    )

    join_result_cudf["right_val"] = (
        join_result_cudf["right_val"].astype(np.float64).fillna(np.nan)
    )

    for col in list(pddf_joined.columns):
        if col.count("_y") > 0:
            join_result[col] = (
                join_result[col].astype(np.float64).fillna(np.nan)
            )
            join_result_cudf[col] = (
                join_result_cudf[col].astype(np.float64).fillna(np.nan)
            )

    # Test dataframe equality (ignore order of rows and columns)
    cdf_result = (
        join_result.to_pandas(nullable_pd_dtype=False)
        .sort_values(list(pddf_joined.columns))
        .reset_index(drop=True)
    )

    pdf_result = pddf_joined.sort_values(
        list(pddf_joined.columns)
    ).reset_index(drop=True)

    assert_eq(cdf_result, pdf_result, check_like=True)

    merge_func_result_cdf = (
        join_result_cudf.to_pandas(nullable_pd_dtype=False)
        .sort_values(list(pddf_joined.columns))
        .reset_index(drop=True)
    )

    assert_eq(merge_func_result_cdf, cdf_result, check_like=True)


def test_dataframe_merge_on_unknown_column():
    np.random.seed(0)

    # Make cuDF
    df_left = DataFrame()
    nelem = 500
    df_left["key1"] = np.random.randint(0, 40, nelem)
    df_left["key2"] = np.random.randint(0, 50, nelem)
    df_left["left_val"] = np.arange(nelem)

    df_right = DataFrame()
    nelem = 500
    df_right["key1"] = np.random.randint(0, 30, nelem)
    df_right["key2"] = np.random.randint(0, 50, nelem)
    df_right["right_val"] = np.arange(nelem)

    with pytest.raises(KeyError) as raises:
        df_left.merge(df_right, on="bad_key", how="left")
    raises.match("bad_key")


def test_dataframe_merge_no_common_column():
    np.random.seed(0)

    # Make cuDF
    df_left = DataFrame()
    nelem = 500
    df_left["key1"] = np.random.randint(0, 40, nelem)
    df_left["key2"] = np.random.randint(0, 50, nelem)
    df_left["left_val"] = np.arange(nelem)

    df_right = DataFrame()
    nelem = 500
    df_right["key3"] = np.random.randint(0, 30, nelem)
    df_right["key4"] = np.random.randint(0, 50, nelem)
    df_right["right_val"] = np.arange(nelem)

    with pytest.raises(ValueError) as raises:
        df_left.merge(df_right, how="left")
    raises.match("No common columns to perform merge on")


def test_dataframe_empty_merge():
    gdf1 = DataFrame({"a": [], "b": []})
    gdf2 = DataFrame({"a": [], "c": []})

    expect = DataFrame({"a": [], "b": [], "c": []})
    got = gdf1.merge(gdf2, how="left", on=["a"])

    assert_eq(expect, got)


def test_dataframe_merge_order():
    gdf1 = DataFrame()
    gdf2 = DataFrame()
    gdf1["id"] = [10, 11]
    gdf1["timestamp"] = [1, 2]
    gdf1["a"] = [3, 4]

    gdf2["id"] = [4, 5]
    gdf2["a"] = [7, 8]

    gdf = gdf1.merge(gdf2, how="left", on=["id", "a"], method="hash")

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df1["id"] = [10, 11]
    df1["timestamp"] = [1, 2]
    df1["a"] = [3, 4]

    df2["id"] = [4, 5]
    df2["a"] = [7, 8]

    df = df1.merge(df2, how="left", on=["id", "a"])
    assert_eq(gdf, df)


@pytest.mark.parametrize(
    "pairs",
    [
        ("", ""),
        ("", "a"),
        ("", "ab"),
        ("", "abc"),
        ("", "b"),
        ("", "bcd"),
        ("", "cde"),
        ("a", "a"),
        ("a", "ab"),
        ("a", "abc"),
        ("a", "b"),
        ("a", "bcd"),
        ("a", "cde"),
        ("ab", "ab"),
        ("ab", "abc"),
        ("ab", "b"),
        ("ab", "bcd"),
        ("ab", "cde"),
        ("abc", "abc"),
        ("abc", "b"),
        ("abc", "bcd"),
        ("abc", "cde"),
        ("b", "b"),
        ("b", "bcd"),
        ("b", "cde"),
        ("bcd", "bcd"),
        ("bcd", "cde"),
        ("cde", "cde"),
    ],
)
@pytest.mark.parametrize("max", [5, 1000])
@pytest.mark.parametrize("rows", [1, 5, 100])
@pytest.mark.parametrize("how", ["left", "inner", "outer"])
def test_dataframe_pairs_of_triples(pairs, max, rows, how):
    np.random.seed(0)

    pdf_left = pd.DataFrame()
    pdf_right = pd.DataFrame()
    for left_column in pairs[0]:
        pdf_left[left_column] = np.random.randint(0, max, rows)
    for right_column in pairs[1]:
        pdf_right[right_column] = np.random.randint(0, max, rows)
    gdf_left = DataFrame.from_pandas(pdf_left)
    gdf_right = DataFrame.from_pandas(pdf_right)
    if not set(pdf_left.columns).intersection(pdf_right.columns):
        with pytest.raises(pd.core.reshape.merge.MergeError) as raises:
            pdf_left.merge(pdf_right)
        raises.match("No common columns to perform merge on")
        with pytest.raises(ValueError) as raises:
            gdf_left.merge(gdf_right)
        raises.match("No common columns to perform merge on")
    elif not [value for value in pdf_left if value in pdf_right]:
        with pytest.raises(pd.core.reshape.merge.MergeError) as raises:
            pdf_left.merge(pdf_right)
        raises.match("No common columns to perform merge on")
        with pytest.raises(ValueError) as raises:
            gdf_left.merge(gdf_right)
        raises.match("No common columns to perform merge on")
    else:
        pdf_result = pdf_left.merge(pdf_right, how=how)
        gdf_result = gdf_left.merge(gdf_right, how=how)
        assert np.array_equal(gdf_result.columns, pdf_result.columns)
        for column in gdf_result:
            gdf_col_result_sorted = gdf_result[column].fillna(-1).sort_values()
            pd_col_result_sorted = pdf_result[column].fillna(-1).sort_values()
            assert np.array_equal(
                gdf_col_result_sorted.to_pandas().values,
                pd_col_result_sorted.values,
            )


def test_safe_merging_with_left_empty():
    import numpy as np
    import pandas as pd

    from cudf import DataFrame

    np.random.seed(0)

    pairs = ("bcd", "b")
    pdf_left = pd.DataFrame()
    pdf_right = pd.DataFrame()
    for left_column in pairs[0]:
        pdf_left[left_column] = np.random.randint(0, 10, 0)
    for right_column in pairs[1]:
        pdf_right[right_column] = np.random.randint(0, 10, 5)
    gdf_left = DataFrame.from_pandas(pdf_left)
    gdf_right = DataFrame.from_pandas(pdf_right)

    pdf_result = pdf_left.merge(pdf_right)
    gdf_result = gdf_left.merge(gdf_right)
    # Simplify test because pandas does not consider empty Index and RangeIndex
    # to be equivalent. TODO: Allow empty Index objects to have equivalence.
    assert len(pdf_result) == len(gdf_result)


@pytest.mark.parametrize("how", ["left", "inner", "outer"])
@pytest.mark.parametrize("left_empty", [True, False])
@pytest.mark.parametrize("right_empty", [True, False])
def test_empty_joins(how, left_empty, right_empty):
    pdf = pd.DataFrame({"x": [1, 2, 3]})

    if left_empty:
        left = pdf.head(0)
    else:
        left = pdf
    if right_empty:
        right = pdf.head(0)
    else:
        right = pdf

    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)

    expected = left.merge(right, how=how)
    result = gleft.merge(gright, how=how)
    assert len(expected) == len(result)


@pytest.mark.xfail(
    reason="left_on/right_on produces undefined results with 0"
    "index and is disabled"
)
def test_merge_left_index_zero():
    left = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=[0, 1, 2, 3, 4, 5])
    right = pd.DataFrame(
        {"y": [10, 20, 30, 6, 5, 4]}, index=[0, 1, 2, 3, 4, 6]
    )
    gleft = DataFrame.from_pandas(left)
    gright = DataFrame.from_pandas(right)
    pd_merge = left.merge(right, left_on="x", right_on="y")
    gd_merge = gleft.merge(gright, left_on="x", right_on="y")

    assert_eq(pd_merge, gd_merge)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True, "right_on": "y"},
        {"right_index": True, "left_on": "x"},
        {"left_on": "x", "right_on": "y"},
        {"left_index": True, "right_index": True},
    ],
)
def test_merge_left_right_index_left_right_on_zero_kwargs(kwargs):
    left = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=[0, 1, 2, 3, 4, 5])
    right = pd.DataFrame(
        {"y": [10, 20, 30, 6, 5, 4]}, index=[0, 1, 2, 3, 4, 6]
    )
    gleft = DataFrame.from_pandas(left)
    gright = DataFrame.from_pandas(right)
    pd_merge = left.merge(right, **kwargs)
    gd_merge = gleft.merge(gright, **kwargs)
    assert_eq(pd_merge, gd_merge)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True, "right_on": "y"},
        {"right_index": True, "left_on": "x"},
        {"left_on": "x", "right_on": "y"},
        {"left_index": True, "right_index": True},
    ],
)
def test_merge_left_right_index_left_right_on_kwargs(kwargs):
    left = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=[1, 2, 3, 4, 5, 6])
    right = pd.DataFrame(
        {"y": [10, 20, 30, 6, 5, 4]}, index=[1, 2, 3, 4, 5, 7]
    )
    gleft = DataFrame.from_pandas(left)
    gright = DataFrame.from_pandas(right)
    pd_merge = left.merge(right, **kwargs)
    gd_merge = gleft.merge(gright, **kwargs)
    assert_eq(pd_merge, gd_merge)


def test_indicator():
    gdf = cudf.DataFrame({"x": [1, 2, 1]})
    gdf.merge(gdf, indicator=False)

    with pytest.raises(NotImplementedError) as info:
        gdf.merge(gdf, indicator=True)

    assert "indicator=False" in str(info.value)


def test_merge_suffixes():
    pdf = cudf.DataFrame({"x": [1, 2, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 1]})
    assert_eq(
        gdf.merge(gdf, suffixes=("left", "right")),
        pdf.merge(pdf, suffixes=("left", "right")),
    )

    with pytest.raises(ValueError) as info:
        gdf.merge(gdf, lsuffix="left", rsuffix="right")

    assert "suffixes=('left', 'right')" in str(info.value)


def test_merge_left_on_right_on():
    left = pd.DataFrame({"xx": [1, 2, 3, 4, 5, 6]})
    right = pd.DataFrame({"xx": [10, 20, 30, 6, 5, 4]})

    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)

    assert_eq(left.merge(right, on="xx"), gleft.merge(gright, on="xx"))

    assert_eq(
        left.merge(right, left_on="xx", right_on="xx"),
        gleft.merge(gright, left_on="xx", right_on="xx"),
    )


def test_merge_on_index_retained():
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3, 4, 5]
    df["b"] = ["a", "b", "c", "d", "e"]
    df.index = [5, 3, 4, 2, 1]

    df2 = cudf.DataFrame()
    df2["a2"] = [1, 2, 3, 4, 5]
    df2["res"] = ["a", "b", "c", "d", "e"]

    pdf = df.to_pandas(nullable_pd_dtype=False)
    pdf2 = df2.to_pandas(nullable_pd_dtype=False)

    gdm = df.merge(df2, left_index=True, right_index=True, how="left")
    pdm = pdf.merge(pdf2, left_index=True, right_index=True, how="left")
    gdm["a2"] = gdm["a2"].astype("float64")
    assert_eq(gdm.sort_index(), pdm.sort_index())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True, "right_on": "y"},
        {"right_index": True, "left_on": "x"},
        {"left_on": "x", "right_on": "y"},
    ],
)
def test_merge_left_right_index_left_right_on_kwargs2(kwargs):
    left = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 20, 30])
    right = pd.DataFrame({"y": [10, 20, 30]}, index=[1, 2, 30])
    gleft = DataFrame.from_pandas(left)
    gright = DataFrame.from_pandas(right)
    gd_merge = gleft.merge(gright, **kwargs)
    pd_merge = left.merge(right, **kwargs)
    if pd_merge.empty:
        assert gd_merge.empty


@pytest.mark.parametrize(
    "hows", [{"how": "inner"}, {"how": "left"}, {"how": "outer"}]
)
@pytest.mark.parametrize(
    "ons",
    [
        {"on": "a"},
        {"on": ["a", "b"]},
        {"on": ["b", "a"]},
        {"on": ["a", "aa", "b"]},
        {"on": ["b", "a", "aa"]},
    ],
)
def test_merge_sort(ons, hows):
    kwargs = {}
    kwargs.update(hows)
    kwargs.update(ons)
    kwargs["sort"] = True
    a = [4, 6, 9, 5, 2, 4, 1, 8, 1]
    b = [9, 8, 7, 8, 3, 9, 7, 9, 2]
    aa = [8, 9, 2, 9, 3, 1, 2, 3, 4]
    left = pd.DataFrame({"a": a, "b": b, "aa": aa})
    right = left.copy(deep=True)

    left.index = [6, 5, 4, 7, 5, 5, 5, 4, 4]
    right.index = [5, 4, 1, 9, 4, 3, 5, 4, 4]

    gleft = DataFrame.from_pandas(left)
    gright = DataFrame.from_pandas(right)
    gd_merge = gleft.merge(gright, **kwargs)

    pd_merge = left.merge(right, **kwargs)
    # require the join keys themselves to be sorted correctly
    # the non-key columns will NOT match pandas ordering
    assert_eq(pd_merge[kwargs["on"]], gd_merge[kwargs["on"]])
    pd_merge = pd_merge.drop(kwargs["on"], axis=1)
    gd_merge = gd_merge.drop(kwargs["on"])
    if not pd_merge.empty:
        # check to make sure the non join key columns are the same
        pd_merge = pd_merge.sort_values(list(pd_merge.columns)).reset_index(
            drop=True
        )
        gd_merge = gd_merge.sort_values(list(gd_merge.columns)).reset_index(
            drop=True
        )

    assert_eq(pd_merge, gd_merge)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_on": ["a"], "left_index": False, "right_index": True},
        {"right_on": ["b"], "left_index": True, "right_index": False},
        {
            "left_on": ["a"],
            "right_on": ["b"],
            "left_index": True,
            "right_index": True,
        },
    ],
)
def test_merge_sort_on_indexes(kwargs):
    left_index = kwargs["left_index"]
    right_index = kwargs["right_index"]
    kwargs["sort"] = True
    a = [4, 6, 9, 5, 2, 4, 1, 8, 1]
    left = pd.DataFrame({"a": a})
    right = pd.DataFrame({"b": a})

    left.index = [6, 5, 4, 7, 5, 5, 5, 4, 4]
    right.index = [5, 4, 1, 9, 4, 3, 5, 4, 4]

    gleft = DataFrame.from_pandas(left)
    gright = DataFrame.from_pandas(right)
    gd_merge = gleft.merge(gright, **kwargs)

    if left_index and right_index:
        check_if_sorted = gd_merge[["a", "b"]].to_pandas(
            nullable_pd_dtype=False
        )
        check_if_sorted.index.name = "index"
        definitely_sorted = check_if_sorted.sort_values(["index", "a", "b"])
        definitely_sorted.index.name = None
        assert_eq(gd_merge, definitely_sorted)
    elif left_index:
        assert gd_merge["b"].is_monotonic
    elif right_index:
        assert gd_merge["a"].is_monotonic


@pytest.mark.parametrize(
    "dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_join_datetimes_index(dtype):
    datetimes = pd.Series(pd.date_range("20010101", "20010102", freq="12h"))
    pdf_lhs = pd.DataFrame(index=[1, 0, 1, 2, 0, 0, 1])
    pdf_rhs = pd.DataFrame({"d": datetimes})
    gdf_lhs = DataFrame.from_pandas(pdf_lhs)
    gdf_rhs = DataFrame.from_pandas(pdf_rhs)

    gdf_rhs["d"] = gdf_rhs["d"].astype(dtype)

    pdf = pdf_lhs.join(pdf_rhs, sort=True)
    gdf = gdf_lhs.join(gdf_rhs, sort=True)

    assert gdf["d"].dtype == np.dtype(dtype)

    assert_eq(pdf, gdf)


def test_join_with_different_names():
    left = pd.DataFrame({"a": [0, 1, 2.0, 3, 4, 5, 9]})
    right = pd.DataFrame({"b": [12, 5, 3, 9.0, 5], "c": [1, 2, 3, 4, 5.0]})
    gleft = DataFrame.from_pandas(left)
    gright = DataFrame.from_pandas(right)
    pd_merge = left.merge(right, how="outer", left_on=["a"], right_on=["b"])
    gd_merge = gleft.merge(gright, how="outer", left_on=["a"], right_on=["b"])
    assert_eq(pd_merge, gd_merge.sort_values(by=["a"]).reset_index(drop=True))


def test_join_same_name_different_order():
    left = pd.DataFrame({"a": [0, 0], "b": [1, 2]})
    right = pd.DataFrame({"a": [1, 2], "b": [0, 0]})
    gleft = DataFrame.from_pandas(left)
    gright = DataFrame.from_pandas(right)
    pd_merge = left.merge(right, left_on=["a", "b"], right_on=["b", "a"])
    gd_merge = gleft.merge(gright, left_on=["a", "b"], right_on=["b", "a"])
    assert_eq(
        pd_merge, gd_merge.sort_values(by=["a_x"]).reset_index(drop=True)
    )


def test_join_empty_table_dtype():
    left = pd.DataFrame({"a": []})
    right = pd.DataFrame({"b": [12, 5, 3, 9.0, 5], "c": [1, 2, 3, 4, 5.0]})
    gleft = DataFrame.from_pandas(left)
    gright = DataFrame.from_pandas(right)
    pd_merge = left.merge(right, how="left", left_on=["a"], right_on=["b"])
    gd_merge = gleft.merge(gright, how="left", left_on=["a"], right_on=["b"])
    assert_eq(pd_merge["a"].dtype, gd_merge["a"].dtype)


@pytest.mark.parametrize("how", ["outer", "inner", "left", "right"])
@pytest.mark.parametrize(
    "column_a",
    [
        (
            pd.Series([None, 1, 2, 3, 4, 5, 6, 7]).astype(np.float),
            pd.Series([8, 9, 10, 11, 12, None, 14, 15]).astype(np.float),
        )
    ],
)
@pytest.mark.parametrize(
    "column_b",
    [
        (
            pd.Series([0, 1, 0, None, 1, 0, 0, 0]).astype(np.float),
            pd.Series([None, 1, 2, 1, 2, 2, 0, 0]).astype(np.float),
        )
    ],
)
@pytest.mark.parametrize(
    "column_c",
    [
        (
            pd.Series(["dog", "cat", "fish", "bug"] * 2),
            pd.Series(["bird", "cat", "mouse", "snake"] * 2),
        ),
        (
            pd.Series(["dog", "cat", "fish", "bug"] * 2).astype("category"),
            pd.Series(["bird", "cat", "mouse", "snake"] * 2).astype(
                "category"
            ),
        ),
    ],
)
def test_join_multi(how, column_a, column_b, column_c):
    index = ["b", "c"]
    df1 = pd.DataFrame()
    df1["a1"] = column_a[0]
    df1["b"] = column_b[0]
    df1["c"] = column_c[0]
    df1 = df1.set_index(index)
    gdf1 = cudf.from_pandas(df1)

    df2 = pd.DataFrame()
    df2["a2"] = column_a[1]
    df2["b"] = column_b[1]
    df2["c"] = column_c[1]
    df2 = df2.set_index(index)
    gdf2 = cudf.from_pandas(df2)

    gdf_result = gdf1.join(gdf2, how=how, sort=True)
    pdf_result = df1.join(df2, how=how, sort=True)

    # Make sure columns are in the same order
    columns = pdf_result.columns.values
    gdf_result = gdf_result[columns]
    pdf_result = pdf_result[columns]

    assert_eq(
        gdf_result.reset_index(drop=True).fillna(-1),
        pdf_result.sort_index().reset_index(drop=True).fillna(-1),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "left_on": ["a", "b"],
            "right_on": ["a", "b"],
            "left_index": False,
            "right_index": False,
        },  # left and right on, no indices
        {
            "left_on": None,
            "right_on": None,
            "left_index": True,
            "right_index": True,
        },  # left_index and right_index, no on
        {
            "left_on": ["a", "b"],
            "right_on": None,
            "left_index": False,
            "right_index": True,
        },  # left on and right_index
        {
            "left_on": None,
            "right_on": ["a", "b"],
            "left_index": True,
            "right_index": False,
        },  # right_on and left_index
    ],
)
def test_merge_multi(kwargs):

    left = DataFrame(
        {
            "a": [1, 2, 3, 4, 3, 5, 6],
            "b": [1, 3, 5, 7, 5, 9, 0],
            "c": ["o", "p", "q", "r", "s", "t", "u"],
            "d": ["v", "w", "x", "y", "z", "1", "2"],
        }
    )
    right = DataFrame(
        {
            "a": [0, 9, 3, 4, 3, 7, 8],
            "b": [2, 4, 5, 7, 5, 6, 8],
            "c": ["a", "b", "c", "d", "e", "f", "g"],
            "d": ["j", "i", "j", "k", "l", "m", "n"],
        }
    )

    if (
        kwargs["left_on"] is not None
        and kwargs["right_on"] is not None
        and kwargs["left_index"] is False
        and kwargs["right_index"] is False
    ):
        left = left.set_index(["c", "d"])
        right = right.set_index(["c", "d"])
    elif (
        kwargs["left_on"] is None
        and kwargs["right_on"] is None
        and kwargs["left_index"] is True
        and kwargs["right_index"] is True
    ):
        left = left.set_index(["a", "b"])
        right = right.set_index(["a", "b"])
    elif kwargs["left_on"] is not None and kwargs["right_index"] is True:
        left = left.set_index(["c", "d"])
        right = right.set_index(["a", "b"])
    elif kwargs["right_on"] is not None and kwargs["left_index"] is True:
        left = left.set_index(["a", "b"])
        right = right.set_index(["c", "d"])

    gleft = left.to_pandas(nullable_pd_dtype=False)
    gright = right.to_pandas(nullable_pd_dtype=False)

    kwargs["sort"] = True
    expect = gleft.merge(gright, **kwargs)
    got = left.merge(right, **kwargs)

    assert_eq(expect.sort_index().index, got.sort_index().index)

    expect.index = range(len(expect))
    got.index = range(len(got))
    expect = expect.sort_values(list(expect.columns))
    got = got.sort_values(list(got.columns))
    expect.index = range(len(expect))
    got.index = range(len(got))

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype_l", INTEGER_TYPES)
@pytest.mark.parametrize("dtype_r", INTEGER_TYPES)
def test_typecast_on_join_int_to_int(dtype_l, dtype_r):
    other_data = ["a", "b", "c"]

    join_data_l = Series([1, 2, 3], dtype=dtype_l)
    join_data_r = Series([1, 2, 4], dtype=dtype_r)

    gdf_l = DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = np.find_common_type([], [np.dtype(dtype_l), np.dtype(dtype_r)])

    exp_join_data = [1, 2]
    exp_other_data = ["a", "b"]
    exp_join_col = Series(exp_join_data, dtype=exp_dtype)

    expect = DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype_l", ["float32", "float64"])
@pytest.mark.parametrize("dtype_r", ["float32", "float64"])
def test_typecast_on_join_float_to_float(dtype_l, dtype_r):
    other_data = ["a", "b", "c", "d", "e", "f"]

    join_data_l = Series([1, 2, 3, 0.9, 4.5, 6], dtype=dtype_l)
    join_data_r = Series([1, 2, 3, 0.9, 4.5, 7], dtype=dtype_r)

    gdf_l = DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = np.find_common_type([], [np.dtype(dtype_l), np.dtype(dtype_r)])

    if dtype_l != dtype_r:
        exp_join_data = [1, 2, 3, 4.5]
        exp_other_data = ["a", "b", "c", "e"]
    else:
        exp_join_data = [1, 2, 3, 0.9, 4.5]
        exp_other_data = ["a", "b", "c", "d", "e"]

    exp_join_col = Series(exp_join_data, dtype=exp_dtype)

    expect = DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype_l", NUMERIC_TYPES)
@pytest.mark.parametrize("dtype_r", NUMERIC_TYPES)
def test_typecast_on_join_mixed_int_float(dtype_l, dtype_r):
    if (
        ("int" in dtype_l or "long" in dtype_l)
        and ("int" in dtype_r or "long" in dtype_r)
    ) or ("float" in dtype_l and "float" in dtype_r):
        pytest.skip("like types not tested in this function")

    other_data = ["a", "b", "c", "d", "e", "f"]

    join_data_l = Series([1, 2, 3, 0.9, 4.5, 6], dtype=dtype_l)
    join_data_r = Series([1, 2, 3, 0.9, 4.5, 7], dtype=dtype_r)

    gdf_l = DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = np.find_common_type([], [np.dtype(dtype_l), np.dtype(dtype_r)])

    exp_join_data = [1, 2, 3]
    exp_other_data = ["a", "b", "c"]
    exp_join_col = Series(exp_join_data, dtype=exp_dtype)

    expect = DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_eq(expect, got)


def test_typecast_on_join_no_float_round():

    other_data = ["a", "b", "c", "d", "e"]

    join_data_l = Series([1, 2, 3, 4, 5], dtype="int8")
    join_data_r = Series([1, 2, 3, 4.01, 4.99], dtype="float32")

    gdf_l = DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = DataFrame({"join_col": join_data_r, "B": other_data})

    exp_join_data = [1, 2, 3, 4, 5]
    exp_Bx = ["a", "b", "c", "d", "e"]
    exp_By = ["a", "b", "c", None, None]
    exp_join_col = Series(exp_join_data, dtype="float32")

    expect = DataFrame(
        {"join_col": exp_join_col, "B_x": exp_Bx, "B_y": exp_By}
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="left")

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtypes",
    [
        (np.dtype("int8"), np.dtype("int16")),
        (np.dtype("int16"), np.dtype("int32")),
        (np.dtype("int32"), np.dtype("int64")),
        (np.dtype("uint8"), np.dtype("uint16")),
        (np.dtype("uint16"), np.dtype("uint32")),
        (np.dtype("uint32"), np.dtype("uint64")),
        (np.dtype("float32"), np.dtype("float64")),
        (np.dtype("int32"), np.dtype("float32")),
        (np.dtype("uint32"), np.dtype("float32")),
    ],
)
def test_typecast_on_join_overflow_unsafe(dtypes):
    dtype_l, dtype_r = dtypes
    if dtype_l.kind in {"i", "u"}:
        dtype_l_max = np.iinfo(dtype_l).max
    elif dtype_l.kind == "f":
        dtype_l_max = np.finfo(dtype_r).max

    lhs = cudf.DataFrame({"a": [1, 2, 3, 4, 5]}, dtype=dtype_l)
    rhs = cudf.DataFrame({"a": [1, 2, 3, 4, dtype_l_max + 1]}, dtype=dtype_r)

    with pytest.warns(
        UserWarning,
        match=(
            f"can't safely cast column"
            f" from right with type {dtype_r} to {dtype_l}"
        ),
    ):
        merged = lhs.merge(rhs, on="a", how="left")  # noqa: F841


@pytest.mark.parametrize(
    "dtype_l",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
@pytest.mark.parametrize(
    "dtype_r",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_on_join_dt_to_dt(dtype_l, dtype_r):
    other_data = ["a", "b", "c", "d", "e"]
    join_data_l = Series(
        ["1991-11-20", "1999-12-31", "2004-12-04", "2015-01-01", "2019-08-15"]
    ).astype(dtype_l)
    join_data_r = Series(
        ["1991-11-20", "1999-12-31", "2004-12-04", "2015-01-01", "2019-08-16"]
    ).astype(dtype_r)

    gdf_l = DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = max(np.dtype(dtype_l), np.dtype(dtype_r))

    exp_join_data = ["1991-11-20", "1999-12-31", "2004-12-04", "2015-01-01"]
    exp_other_data = ["a", "b", "c", "d"]
    exp_join_col = Series(exp_join_data, dtype=exp_dtype)

    expect = DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype_l", ["category", "str", "int32", "float32"])
@pytest.mark.parametrize("dtype_r", ["category", "str", "int32", "float32"])
def test_typecast_on_join_categorical(dtype_l, dtype_r):
    if not (dtype_l == "category" or dtype_r == "category"):
        pytest.skip("at least one side must be category for this set of tests")
    if dtype_l == "category" and dtype_r == "category":
        pytest.skip("Can't determine which categorical to use")

    other_data = ["a", "b", "c", "d", "e"]
    join_data_l = Series([1, 2, 3, 4, 5], dtype=dtype_l)
    join_data_r = Series([1, 2, 3, 4, 6], dtype=dtype_r)
    if dtype_l == "category":
        exp_dtype = join_data_l.dtype.categories.dtype
    elif dtype_r == "category":
        exp_dtype = join_data_r.dtype.categories.dtype

    gdf_l = DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = DataFrame({"join_col": join_data_r, "B": other_data})

    exp_join_data = [1, 2, 3, 4]
    exp_other_data = ["a", "b", "c", "d"]
    exp_join_col = Series(exp_join_data, dtype=exp_dtype)

    expect = DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")
    assert_eq(expect, got)


@pytest.mark.parametrize(
    ("lhs", "rhs"),
    [
        (["a", "b"], ["a"]),
        (["a"], ["a", "b"]),
        (["a", "b"], ["b"]),
        (["b"], ["a", "b"]),
        (["a"], ["a"]),
    ],
)
@pytest.mark.parametrize("how", ["left", "right", "outer", "inner"])
@pytest.mark.parametrize("level", ["a", "b", 0, 1])
def test_index_join(lhs, rhs, how, level):
    l_pdf = pd.DataFrame({"a": [2, 3, 1, 4], "b": [3, 7, 8, 1]})
    r_pdf = pd.DataFrame({"a": [1, 5, 4, 0], "b": [3, 9, 8, 4]})
    l_df = DataFrame.from_pandas(l_pdf)
    r_df = DataFrame.from_pandas(r_pdf)
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index

    expected = (
        p_lhs.join(p_rhs, level=level, how=how)
        .to_frame(index=False)
        .sort_values(by=lhs)
        .reset_index(drop=True)
    )
    got = (
        g_lhs.join(g_rhs, level=level, how=how)
        .to_frame(index=False)
        .sort_values(by=lhs)
        .reset_index(drop=True)
    )

    assert_eq(expected, got)


def test_index_join_corner_cases():
    l_pdf = pd.DataFrame({"a": [2, 3, 1, 4], "b": [3, 7, 8, 1]})
    r_pdf = pd.DataFrame(
        {"a": [1, 5, 4, 0], "b": [3, 9, 8, 4], "c": [2, 3, 6, 0]}
    )
    l_df = DataFrame.from_pandas(l_pdf)
    r_df = DataFrame.from_pandas(r_pdf)

    # Join when column name doesn't match with level
    lhs = ["a", "b"]
    # level and rhs don't match
    rhs = ["c"]
    level = "b"
    how = "outer"
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    expected = (
        p_lhs.join(p_rhs, level=level, how=how)
        .to_frame(index=False)
        .sort_values(by=lhs)
        .reset_index(drop=True)
    )
    got = (
        g_lhs.join(g_rhs, level=level, how=how)
        .to_frame(index=False)
        .sort_values(by=lhs)
        .reset_index(drop=True)
    )

    assert_eq(expected, got)

    # sort is supported only in case of two non-MultiIndex join
    # Join when column name doesn't match with level
    lhs = ["a"]
    # level and rhs don't match
    rhs = ["a"]
    level = "b"
    how = "left"
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    expected = p_lhs.join(p_rhs, how=how, sort=True)
    got = g_lhs.join(g_rhs, how=how, sort=True)

    assert_eq(expected, got)

    # Pandas Index.join on categorical column returns generic column
    # but cudf will be returning a categorical column itself.
    lhs = ["a", "b"]
    rhs = ["a"]
    level = "a"
    how = "inner"
    l_df["a"] = l_df["a"].astype("category")
    r_df["a"] = r_df["a"].astype("category")
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    expected = (
        p_lhs.join(p_rhs, level=level, how=how)
        .to_frame(index=False)
        .sort_values(by=lhs)
        .reset_index(drop=True)
    )
    got = (
        g_lhs.join(g_rhs, level=level, how=how)
        .to_frame(index=False)
        .sort_values(by=lhs)
        .reset_index(drop=True)
    )

    got["a"] = got["a"].astype(expected["a"].dtype)

    assert_eq(expected, got)


def test_index_join_exception_cases():
    l_df = DataFrame({"a": [2, 3, 1, 4], "b": [3, 7, 8, 1]})
    r_df = DataFrame({"a": [1, 5, 4, 0], "b": [3, 9, 8, 4], "c": [2, 3, 6, 0]})

    # Join between two MultiIndex
    lhs = ["a", "b"]
    rhs = ["a", "c"]
    level = "a"
    how = "outer"
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index

    with pytest.raises(TypeError):
        g_lhs.join(g_rhs, level=level, how=how)

    # Improper level value, level should be an int or scalar value
    level = ["a"]
    rhs = ["a"]
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    with pytest.raises(ValueError):
        g_lhs.join(g_rhs, level=level, how=how)


def test_typecast_on_join_indexes():
    join_data_l = Series([1, 2, 3, 4, 5], dtype="int8")
    join_data_r = Series([1, 2, 3, 4, 6], dtype="int32")
    other_data = ["a", "b", "c", "d", "e"]

    gdf_l = DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = DataFrame({"join_col": join_data_r, "B": other_data})

    gdf_l = gdf_l.set_index("join_col")
    gdf_r = gdf_r.set_index("join_col")

    exp_join_data = [1, 2, 3, 4]
    exp_other_data = ["a", "b", "c", "d"]

    expect = DataFrame(
        {
            "join_col": exp_join_data,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )
    expect = expect.set_index("join_col")

    got = gdf_l.join(gdf_r, how="inner", lsuffix="_x", rsuffix="_y")

    assert_eq(expect, got)


def test_typecast_on_join_multiindices():
    join_data_l_0 = Series([1, 2, 3, 4, 5], dtype="int8")
    join_data_l_1 = Series([2, 3, 4.1, 5.9, 6], dtype="float32")
    join_data_l_2 = Series([7, 8, 9, 0, 1], dtype="float32")

    join_data_r_0 = Series([1, 2, 3, 4, 5], dtype="int32")
    join_data_r_1 = Series([2, 3, 4, 5, 6], dtype="int32")
    join_data_r_2 = Series([7, 8, 9, 0, 0], dtype="float64")

    other_data = ["a", "b", "c", "d", "e"]

    gdf_l = DataFrame(
        {
            "join_col_0": join_data_l_0,
            "join_col_1": join_data_l_1,
            "join_col_2": join_data_l_2,
            "B": other_data,
        }
    )
    gdf_r = DataFrame(
        {
            "join_col_0": join_data_r_0,
            "join_col_1": join_data_r_1,
            "join_col_2": join_data_r_2,
            "B": other_data,
        }
    )

    gdf_l = gdf_l.set_index(["join_col_0", "join_col_1", "join_col_2"])
    gdf_r = gdf_r.set_index(["join_col_0", "join_col_1", "join_col_2"])

    exp_join_data_0 = Series([1, 2], dtype="int32")
    exp_join_data_1 = Series([2, 3], dtype="float64")
    exp_join_data_2 = Series([7, 8], dtype="float64")
    exp_other_data = Series(["a", "b"])

    expect = DataFrame(
        {
            "join_col_0": exp_join_data_0,
            "join_col_1": exp_join_data_1,
            "join_col_2": exp_join_data_2,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )
    expect = expect.set_index(["join_col_0", "join_col_1", "join_col_2"])
    got = gdf_l.join(gdf_r, how="inner", lsuffix="_x", rsuffix="_y")

    assert_eq(expect, got)


def test_typecast_on_join_indexes_matching_categorical():
    join_data_l = Series(["a", "b", "c", "d", "e"], dtype="category")
    join_data_r = Series(["a", "b", "c", "d", "e"], dtype="str")
    other_data = [1, 2, 3, 4, 5]

    gdf_l = DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = DataFrame({"join_col": join_data_r, "B": other_data})

    gdf_l = gdf_l.set_index("join_col")
    gdf_r = gdf_r.set_index("join_col")

    exp_join_data = ["a", "b", "c", "d", "e"]
    exp_other_data = [1, 2, 3, 4, 5]

    expect = DataFrame(
        {
            "join_col": exp_join_data,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )
    expect = expect.set_index("join_col")
    got = gdf_l.join(gdf_r, how="inner", lsuffix="_x", rsuffix="_y")

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "lhs",
    [
        cudf.Series([1, 2, 3], name="a"),
        cudf.DataFrame({"a": [2, 3, 4], "c": [4, 5, 6]}),
    ],
)
@pytest.mark.parametrize(
    "rhs",
    [
        cudf.Series([1, 2, 3], name="b"),
        cudf.DataFrame({"b": [2, 3, 4], "c": [4, 5, 6]}),
    ],
)
@pytest.mark.parametrize(
    "how", ["left", "inner", "outer", "leftanti", "leftsemi"]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_on": "a", "right_on": "b"},
        {"left_index": True, "right_on": "b"},
        {"left_on": "a", "right_index": True},
        {"left_index": True, "right_index": True},
        {
            "left_on": "a",
            "right_on": "b",
            "left_index": True,
            "right_index": True,
        },
    ],
)
def test_series_dataframe_mixed_merging(lhs, rhs, how, kwargs):

    if how in ("leftsemi", "leftanti") and (
        kwargs.get("left_index") or kwargs.get("right_index")
    ):
        pytest.skip("Index joins not compatible with leftsemi and leftanti")

    check_lhs = lhs.copy()
    check_rhs = rhs.copy()
    if isinstance(lhs, Series):
        check_lhs = lhs.to_frame()
    if isinstance(rhs, Series):
        check_rhs = rhs.to_frame()

    expect = check_lhs.merge(check_rhs, how=how, **kwargs)
    got = lhs.merge(rhs, how=how, **kwargs)

    assert_eq(expect, got)
