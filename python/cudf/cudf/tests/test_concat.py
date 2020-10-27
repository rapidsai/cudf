# Copyright (c) 2018-2020, NVIDIA CORPORATION.

import re

import numpy as np
import pandas as pd
import pytest

import cudf as gd
from cudf.tests.utils import assert_eq, assert_exceptions_equal
from cudf.utils.dtypes import is_categorical_dtype


def make_frames(index=None, nulls="none"):
    df = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": list("abcde") * 2,
        }
    )
    df.z = df.z.astype("category")
    df2 = pd.DataFrame(
        {
            "x": range(10, 20),
            "y": list(map(float, range(10, 20))),
            "z": list("edcba") * 2,
        }
    )
    df2.z = df2.z.astype("category")
    if nulls == "all":
        df.y = np.full_like(df.y, np.nan)
        df2.y = np.full_like(df2.y, np.nan)
    if nulls == "some":
        mask = np.arange(10)
        np.random.shuffle(mask)
        mask = mask[:5]
        df.y.loc[mask] = np.nan
        df2.y.loc[mask] = np.nan
    gdf = gd.DataFrame.from_pandas(df)
    gdf2 = gd.DataFrame.from_pandas(df2)
    if index:
        df = df.set_index(index)
        df2 = df2.set_index(index)
        gdf = gdf.set_index(index)
        gdf2 = gdf2.set_index(index)
    return df, df2, gdf, gdf2


@pytest.mark.parametrize("nulls", ["none", "some", "all"])
@pytest.mark.parametrize("index", [False, "z", "y"])
@pytest.mark.parametrize("axis", [0, "index"])
def test_concat_dataframe(index, nulls, axis):
    if index == "y" and nulls in ("some", "all"):
        pytest.skip("nulls in columns, dont index")
    df, df2, gdf, gdf2 = make_frames(index, nulls=nulls)
    # Make empty frame
    gdf_empty1 = gdf2[:0]
    assert len(gdf_empty1) == 0
    df_empty1 = gdf_empty1.to_pandas()

    # DataFrame
    res = gd.concat([gdf, gdf2, gdf, gdf_empty1], axis=axis).to_pandas()
    sol = pd.concat([df, df2, df, df_empty1], axis=axis)
    assert_eq(res, sol, check_names=False, check_categorical=False)

    # Series
    for c in [i for i in ("x", "y", "z") if i != index]:
        res = gd.concat([gdf[c], gdf2[c], gdf[c]], axis=axis).to_pandas()
        sol = pd.concat([df[c], df2[c], df[c]], axis=axis)
        assert_eq(res, sol, check_names=False, check_categorical=False)

    # Index
    res = gd.concat([gdf.index, gdf2.index], axis=axis).to_pandas()
    sol = df.index.append(df2.index)
    assert_eq(res, sol, check_names=False, check_categorical=False)


@pytest.mark.parametrize(
    "values",
    [["foo", "bar"], [1.0, 2.0], pd.Series(["one", "two"], dtype="category")],
)
def test_concat_all_nulls(values):
    pa = pd.Series(values)
    pb = pd.Series([None])
    ps = pd.concat([pa, pb])

    ga = gd.Series(values)
    gb = gd.Series([None])
    gs = gd.concat([ga, gb])

    assert_eq(ps, gs, check_dtype=False, check_categorical=False)


def test_concat_errors():
    df, df2, gdf, gdf2 = make_frames()

    # No objs
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=gd.concat,
        lfunc_args_and_kwargs=([], {"objs": []}),
        rfunc_args_and_kwargs=([], {"objs": []}),
    )

    # All None
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=gd.concat,
        lfunc_args_and_kwargs=([], {"objs": [None, None]}),
        rfunc_args_and_kwargs=([], {"objs": [None, None]}),
    )

    # Mismatched types
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=gd.concat,
        lfunc_args_and_kwargs=([], {"objs": [df, df.index, df.x]}),
        rfunc_args_and_kwargs=([], {"objs": [gdf, gdf.index, gdf.x]}),
        expected_error_message=re.escape(
            "`concat` cannot concatenate objects of "
            "types: ['DataFrame', 'RangeIndex', 'Series']."
        ),
    )

    # Unknown type
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=gd.concat,
        lfunc_args_and_kwargs=([], {"objs": ["bar", "foo"]}),
        rfunc_args_and_kwargs=([], {"objs": ["bar", "foo"]}),
        expected_error_message=re.escape(
            "cannot concatenate object of type <class 'str'>"
        ),
    )

    # Mismatched index dtypes
    gdf3 = gdf2.copy()
    del gdf3["z"]
    gdf4 = gdf2.set_index("z")

    with pytest.raises(ValueError, match="All columns must be the same type"):
        gd.concat([gdf3, gdf4])

    # Bad axis value
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=gd.concat,
        lfunc_args_and_kwargs=(
            [],
            {"objs": [gdf.to_pandas(), gdf2.to_pandas()], "axis": "bad_value"},
        ),
        rfunc_args_and_kwargs=([], {"objs": [gdf, gdf2], "axis": "bad_value"}),
        expected_error_message=re.escape(
            '`axis` must be 0 / "index"' ' or 1 / "columns", got: None'
        ),
    )


def test_concat_misordered_columns():
    df, df2, gdf, gdf2 = make_frames(False)
    gdf2 = gdf2[["z", "x", "y"]]
    df2 = df2[["z", "x", "y"]]

    res = gd.concat([gdf, gdf2]).to_pandas()
    sol = pd.concat([df, df2], sort=False)

    assert_eq(res, sol, check_names=False, check_categorical=False)


@pytest.mark.parametrize("axis", [1, "columns"])
def test_concat_columns(axis):
    pdf1 = pd.DataFrame(np.random.randint(10, size=(5, 3)), columns=[1, 2, 3])
    pdf2 = pd.DataFrame(
        np.random.randint(10, size=(5, 4)), columns=[4, 5, 6, 7]
    )
    gdf1 = gd.from_pandas(pdf1)
    gdf2 = gd.from_pandas(pdf2)

    expect = pd.concat([pdf1, pdf2], axis=axis)
    got = gd.concat([gdf1, gdf2], axis=axis)

    assert_eq(expect, got)


def test_concat_multiindex_dataframe():
    gdf = gd.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg.iloc[:, :1]
    pdg2 = pdg.iloc[:, 1:]
    gdg1 = gd.from_pandas(pdg1)
    gdg2 = gd.from_pandas(pdg2)
    assert_eq(
        gd.concat([gdg1, gdg2]).astype("float64"), pd.concat([pdg1, pdg2])
    )
    assert_eq(gd.concat([gdg1, gdg2], axis=1), pd.concat([pdg1, pdg2], axis=1))


def test_concat_multiindex_series():
    gdf = gd.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg["y"]
    pdg2 = pdg["z"]
    gdg1 = gd.from_pandas(pdg1)
    gdg2 = gd.from_pandas(pdg2)
    assert_eq(gd.concat([gdg1, gdg2]), pd.concat([pdg1, pdg2]))
    assert_eq(gd.concat([gdg1, gdg2], axis=1), pd.concat([pdg1, pdg2], axis=1))


def test_concat_multiindex_dataframe_and_series():
    gdf = gd.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg[["y", "z"]]
    pdg2 = pdg["z"]
    pdg2.name = "a"
    gdg1 = gd.from_pandas(pdg1)
    gdg2 = gd.from_pandas(pdg2)
    assert_eq(gd.concat([gdg1, gdg2], axis=1), pd.concat([pdg1, pdg2], axis=1))


def test_concat_multiindex_series_and_dataframe():
    gdf = gd.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg["z"]
    pdg2 = pdg[["y", "z"]]
    pdg1.name = "a"
    gdg1 = gd.from_pandas(pdg1)
    gdg2 = gd.from_pandas(pdg2)
    assert_eq(gd.concat([gdg1, gdg2], axis=1), pd.concat([pdg1, pdg2], axis=1))


@pytest.mark.parametrize("myindex", ["a", "b"])
def test_concat_string_index_name(myindex):
    # GH-Issue #3420
    data = {"a": [123, 456], "b": ["s1", "s2"]}
    df1 = gd.DataFrame(data).set_index(myindex)
    df2 = df1.copy()
    df3 = gd.concat([df1, df2])

    assert df3.index.name == myindex


def test_pandas_concat_compatibility_axis1():
    d1 = gd.datasets.randomdata(
        3, dtypes={"a": float, "ind": float}
    ).set_index("ind")
    d2 = gd.datasets.randomdata(
        3, dtypes={"b": float, "ind": float}
    ).set_index("ind")
    d3 = gd.datasets.randomdata(
        3, dtypes={"c": float, "ind": float}
    ).set_index("ind")
    d4 = gd.datasets.randomdata(
        3, dtypes={"d": float, "ind": float}
    ).set_index("ind")
    d5 = gd.datasets.randomdata(
        3, dtypes={"e": float, "ind": float}
    ).set_index("ind")

    pd1 = d1.to_pandas()
    pd2 = d2.to_pandas()
    pd3 = d3.to_pandas()
    pd4 = d4.to_pandas()
    pd5 = d5.to_pandas()

    expect = pd.concat([pd1, pd2, pd3, pd4, pd5], axis=1)
    got = gd.concat([d1, d2, d3, d4, d5], axis=1)

    assert_eq(got, expect)


@pytest.mark.parametrize("index", [[0, 1, 2], [2, 1, 0], [5, 9, 10]])
@pytest.mark.parametrize("names", [False, (0, 1)])
@pytest.mark.parametrize(
    "data",
    [
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", "b", "c"], ["XX", "YY", "ZZ"]),
    ],
)
def test_pandas_concat_compatibility_axis1_overlap(index, names, data):
    s1 = gd.Series(data[0], index=[0, 1, 2])
    s2 = gd.Series(data[1], index=index)
    if names:
        s1.name = names[0]
        s2.name = names[1]
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    got = gd.concat([s1, s2], axis=1)
    expect = pd.concat([ps1, ps2], axis=1)

    assert_eq(got, expect)


def test_pandas_concat_compatibility_axis1_eq_index():
    s1 = gd.Series(["a", "b", "c"], index=[0, 1, 2])
    s2 = gd.Series(["a", "b", "c"], index=[1, 1, 1])
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()

    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=gd.concat,
        lfunc_args_and_kwargs=([], {"objs": [ps1, ps2], "axis": 1}),
        rfunc_args_and_kwargs=([], {"objs": [s1, s2], "axis": 1}),
    )


def test_concat_duplicate_columns():
    cdf = gd.DataFrame(
        {
            "id4": 4 * list(range(6)),
            "id5": 4 * list(reversed(range(6))),
            "v3": 6 * list(range(4)),
        }
    )
    cdf_std = cdf.groupby(["id4", "id5"])[["v3"]].std()
    cdf_med = cdf.groupby(["id4", "id5"])[["v3"]].quantile(q=0.5)
    with pytest.raises(NotImplementedError):
        gd.concat([cdf_med, cdf_std], axis=1)


def test_concat_mixed_input():
    pdf1 = pd.DataFrame({"a": [10, 20, 30]})
    pdf2 = pd.DataFrame({"a": [11, 22, 33]})

    gdf1 = gd.from_pandas(pdf1)
    gdf2 = gd.from_pandas(pdf2)

    assert_eq(
        pd.concat([pdf1, None, pdf2, None]),
        gd.concat([gdf1, None, gdf2, None]),
    )
    assert_eq(pd.concat([pdf1, None]), gd.concat([gdf1, None]))
    assert_eq(pd.concat([None, pdf2]), gd.concat([None, gdf2]))
    assert_eq(pd.concat([None, pdf2, pdf1]), gd.concat([None, gdf2, gdf1]))


@pytest.mark.parametrize(
    "objs",
    [
        [pd.Series([1, 2, 3]), pd.DataFrame({"a": [1, 2]})],
        [pd.Series([1, 2, 3]), pd.DataFrame({"a": []})],
        [pd.Series([]), pd.DataFrame({"a": []})],
        [pd.Series([]), pd.DataFrame({"a": [1, 2]})],
        [pd.Series([1, 2, 3.0, 1.2], name="abc"), pd.DataFrame({"a": [1, 2]})],
        [
            pd.Series(
                [1, 2, 3.0, 1.2], name="abc", index=[100, 110, 120, 130]
            ),
            pd.DataFrame({"a": [1, 2]}),
        ],
        [
            pd.Series(
                [1, 2, 3.0, 1.2], name="abc", index=["a", "b", "c", "d"]
            ),
            pd.DataFrame({"a": [1, 2]}, index=["a", "b"]),
        ],
        [
            pd.Series(
                [1, 2, 3.0, 1.2, 8, 100],
                name="New name",
                index=["a", "b", "c", "d", "e", "f"],
            ),
            pd.DataFrame(
                {"a": [1, 2, 4, 10, 11, 12]},
                index=["a", "b", "c", "d", "e", "f"],
            ),
        ],
        [
            pd.Series(
                [1, 2, 3.0, 1.2, 8, 100],
                name="New name",
                index=["a", "b", "c", "d", "e", "f"],
            ),
            pd.DataFrame(
                {"a": [1, 2, 4, 10, 11, 12]},
                index=["a", "b", "c", "d", "e", "f"],
            ),
        ]
        * 7,
    ],
)
def test_concat_series_dataframe_input(objs):
    pd_objs = objs
    gd_objs = [gd.from_pandas(obj) for obj in objs]

    expected = pd.concat(pd_objs)
    actual = gd.concat(gd_objs)

    assert_eq(
        expected.fillna(-1),
        actual.fillna(-1),
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [
            pd.Series(["a", "b", "c", "d"]),
            pd.Series(["1", "2", "3", "4"]),
            pd.DataFrame({"first col": ["10", "11", "12", "13"]}),
        ],
        [
            pd.Series(["a", "b", "c", "d"]),
            pd.Series(["1", "2", "3", "4"]),
            pd.DataFrame(
                {
                    "first col": ["10", "11", "12", "13"],
                    "second col": ["a", "b", "c", "d"],
                }
            ),
        ],
        [
            pd.Series(["a", "b", "c"]),
            pd.Series(["1", "2", "3", "4"]),
            pd.DataFrame(
                {
                    "first col": ["10", "11", "12", "13"],
                    "second col": ["a", "b", "c", "d"],
                }
            ),
        ],
    ],
)
def test_concat_series_dataframe_input_str(objs):
    pd_objs = objs
    gd_objs = [gd.from_pandas(obj) for obj in objs]

    expected = pd.concat(pd_objs)
    actual = gd.concat(gd_objs)
    assert_eq(expected, actual, check_dtype=False, check_index_type=False)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame(
            {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
        ),
        pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[200]),
        pd.DataFrame([], index=[100]),
        pd.DataFrame({"cat": pd.Series(["one", "two"], dtype="category")}),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()],
        [
            pd.DataFrame(
                {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        ],
        [
            pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
            pd.DataFrame({"l": [10]}),
            pd.DataFrame({"l": [10]}, index=[200]),
            pd.DataFrame(
                {"cat": pd.Series(["two", "three"], dtype="category")}
            ),
        ],
        [
            pd.DataFrame([]),
            pd.DataFrame([], index=[100]),
            pd.DataFrame(
                {"cat": pd.Series(["two", "three"], dtype="category")}
            ),
        ],
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
def test_concat_empty_dataframes(df, other, ignore_index):
    other_pd = [df] + other

    gdf = gd.from_pandas(df)
    other_gd = [gdf] + [gd.from_pandas(o) for o in other]

    expected = pd.concat(other_pd, ignore_index=ignore_index)
    actual = gd.concat(other_gd, ignore_index=ignore_index)
    if expected.shape != df.shape:
        for key, col in actual[actual.columns].iteritems():
            if is_categorical_dtype(col.dtype):
                expected[key] = expected[key].fillna("-1")
                actual[key] = col.astype("str").fillna("-1")
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(
            expected, actual, check_index_type=False if gdf.empty else True
        )


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("axis", [0, "index"])
@pytest.mark.parametrize(
    "data",
    [
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", "b", "c"], ["XX", "YY", "ZZ"]),
    ],
)
def test_concat_empty_and_nonempty_series(ignore_index, data, axis):
    s1 = gd.Series()
    s2 = gd.Series(data[0])
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    got = gd.concat([s1, s2], axis=axis, ignore_index=ignore_index)
    expect = pd.concat([ps1, ps2], axis=axis, ignore_index=ignore_index)

    assert_eq(got, expect)


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("axis", [0, "index"])
def test_concat_two_empty_series(ignore_index, axis):
    s1 = gd.Series()
    s2 = gd.Series()
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    got = gd.concat([s1, s2], axis=axis, ignore_index=ignore_index)
    expect = pd.concat([ps1, ps2], axis=axis, ignore_index=ignore_index)

    assert_eq(got, expect)


@pytest.mark.parametrize(
    "df1,df2",
    [
        (
            gd.DataFrame({"k1": [0, 1], "k2": [2, 3], "v1": [4, 5]}),
            gd.DataFrame({"k1": [1, 0], "k2": [3, 2], "v2": [6, 7]}),
        ),
        (
            gd.DataFrame({"k1": [0, 1], "k2": [2, 3], "v1": [4, 5]}),
            gd.DataFrame({"k1": [0, 1], "k2": [3, 2], "v2": [6, 7]}),
        ),
    ],
)
def test_concat_dataframe_with_multiIndex(df1, df2):
    gdf1 = df1
    gdf1 = gdf1.set_index(["k1", "k2"])

    gdf2 = df2
    gdf2 = gdf2.set_index(["k1", "k2"])

    pdf1 = gdf1.to_pandas()
    pdf2 = gdf2.to_pandas()

    expected = gd.concat([gdf1, gdf2], axis=1)
    actual = pd.concat([pdf1, pdf2], axis=1)

    assert_eq(expected, actual)
