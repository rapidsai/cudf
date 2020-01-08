# Copyright (c) 2018, NVIDIA CORPORATION.

from timeit import default_timer as timer

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import DataFrame
from cudf.tests.utils import assert_eq


def make_params():
    np.random.seed(0)

    hows = "left,inner,outer,right".split(",")
    methods = "hash,sort".split(",")

    # Test specific cases (1)
    aa = [0, 0, 4, 5, 5]
    bb = [0, 0, 2, 3, 5]
    for how in hows:
        if how in ["left", "inner", "right"]:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield (aa, bb, how, "sort")

    # Test specific cases (2)
    aa = [0, 0, 1, 2, 3]
    bb = [0, 1, 2, 2, 3]
    for how in hows:
        if how in ["left", "inner", "right"]:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield (aa, bb, how, "sort")

    # Test large random integer inputs
    aa = np.random.randint(0, 50, 100)
    bb = np.random.randint(0, 50, 100)
    for how in hows:
        if how in ["left", "inner", "right"]:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield (aa, bb, how, "sort")

    # Test floating point inputs
    aa = np.random.random(50)
    bb = np.random.random(50)
    for how in hows:
        if how in ["left", "inner", "right"]:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield (aa, bb, how, "sort")


@pytest.mark.parametrize("aa,bb,how,method", make_params())
def test_dataframe_join_how(aa, bb, how, method):
    df = DataFrame()
    df["a"] = aa
    df["b"] = bb

    def work_pandas(df):
        ts = timer()
        df1 = df.set_index("a")
        df2 = df.set_index("b")
        joined = df1.join(df2, how=how, sort=True)
        te = timer()
        print("timing", type(df), te - ts)
        return joined

    def work_gdf(df):
        ts = timer()
        df1 = df.set_index("a")
        df2 = df.set_index("b")
        joined = df1.join(df2, how=how, sort=True, method=method)
        te = timer()
        print("timing", type(df), te - ts)
        return joined

    expect = work_pandas(df.to_pandas())
    got = work_gdf(df)
    expecto = expect.copy()
    goto = got.copy()

    # Type conversion to handle NoneType
    expectb = expect.b
    expecta = expect.a
    gotb = got.b
    gota = got.a
    del got["b"]
    got.insert(len(got._data), "b", gotb.astype(np.float64).fillna(np.nan))
    del got["a"]
    got.insert(len(got._data), "a", gota.astype(np.float64).fillna(np.nan))
    expect.drop(["b"], axis=1)
    expect["b"] = expectb.astype(np.float64).fillna(np.nan)
    expect.drop(["a"], axis=1)
    expect["a"] = expecta.astype(np.float64).fillna(np.nan)

    assert got.index.name is None

    assert list(expect.columns) == list(got.columns)
    # test disabled until libgdf sort join gets updated with new api
    if method == "hash":
        assert np.all(expect.index.values == got.index.values)
        if how != "outer":
            # Newly introduced ambiguous ValueError thrown when
            # an index and column have the same name. Rename the
            # index so sorts work.
            # TODO: What is the less hacky way?
            expect.index.name = "bob"
            got.index.name = "mary"
            pd.util.testing.assert_frame_equal(
                got.to_pandas().sort_values(["b", "a"]).reset_index(drop=True),
                expect.sort_values(["b", "a"]).reset_index(drop=True),
            )
        # if(how=='right'):
        #     _sorted_check_series(expect['a'], expect['b'],
        #                          got['a'], got['b'])
        # else:
        #     _sorted_check_series(expect['b'], expect['a'], got['b'],
        #                          got['a'])
        else:
            _check_series(expecto["b"].fillna(-1), goto["b"].fillna(-1))
            _check_series(expecto["a"].fillna(-1), goto["a"].fillna(-1))


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
    assert np.all(expect.index.values == got.index.values)
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
    expect = lhs.to_pandas().join(rhs.to_pandas())

    # Note: pandas make a object Index after joining
    pd.util.testing.assert_frame_equal(
        got.sort_values(by="b")
        .to_pandas()
        .sort_index()
        .reset_index(drop=True),
        expect.reset_index(drop=True),
    )

    # Just do some rough checking here.
    assert list(got.columns) == ["b", "c"]
    assert len(got) > 0
    assert set(got.index.values) & set("abc")
    assert set(got["b"]) & set(bb)
    assert set(got["c"]) & set(cc)


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

    got = join_gdf.to_pandas()
    expect = join_pdf.fillna(-1)  # note: cudf join doesn't mask NA

    # cudf creates the columns in different order than pandas for right join
    if how == "right":
        got = got[["data_col_left", "data_col_right"]]

    expect.data_col_right = expect.data_col_right.astype(np.int64)
    expect.data_col_left = expect.data_col_left.astype(np.int64)
    # Expect has the wrong index type. Quick fix to get index type working
    # again I think this implies that CategoricalIndex.to_pandas() is not
    # working correctly, since the below corrects it. Remove this line for
    # an annoying error. TODO: Make CategoricalIndex.to_pandas() work
    # correctly for the below case.
    # Error:
    # AssertionError: Categorical Expected type <class
    # 'pandas.core.arrays.categorical.Categorical'>, found <class
    # 'numpy.ndarray'> instead
    expect.index = pd.Categorical(expect.index)
    pd.util.testing.assert_frame_equal(
        got,
        expect,
        check_names=False,
        check_index_type=False,
        # For inner joins, pandas returns
        # weird categories.
        check_categorical=how != "inner",
    )
    assert list(got.index) == list(expect.index)


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
    pddf_left = df_left.to_pandas()
    pddf_right = df_right.to_pandas()

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
        join_result.to_pandas()
        .sort_values(list(pddf_joined.columns))
        .reset_index(drop=True)
    )

    pdf_result = pddf_joined.sort_values(
        list(pddf_joined.columns)
    ).reset_index(drop=True)

    pd.util.testing.assert_frame_equal(cdf_result, pdf_result, check_like=True)

    merge_func_result_cdf = (
        join_result_cudf.to_pandas()
        .sort_values(list(pddf_joined.columns))
        .reset_index(drop=True)
    )

    pd.util.testing.assert_frame_equal(
        merge_func_result_cdf, cdf_result, check_like=True
    )


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
    from cudf import DataFrame
    import pandas as pd

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

    pdf = df.to_pandas()
    pdf2 = df2.to_pandas()

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
    "kwargs",
    [{"on": "k1"}, {"on": "k2"}, {"on": "k3"}, {"on": "k4"}, {"on": "k5"}],
)
def test_merge_sort(kwargs, hows):
    kwargs.update(hows)
    kwargs["sort"] = True
    d = range(3)
    left = pd.DataFrame({"k2": d, "k1": d, "k4": d, "k3": d, "k5": d})
    right = pd.DataFrame({"k1": d, "k4": d, "k2": d, "k3": d, "k5": d})
    gleft = DataFrame.from_pandas(left)
    gright = DataFrame.from_pandas(right)
    gd_merge = gleft.merge(gright, **kwargs)
    pd_merge = left.merge(right, **kwargs)
    if pd_merge.empty:
        assert gd_merge.empty


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
