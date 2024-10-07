# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import warnings
from contextlib import contextmanager
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_220
from cudf.core.dtypes import Decimal32Dtype, Decimal64Dtype, Decimal128Dtype
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@contextmanager
def _hide_concat_empty_dtype_warning():
    with warnings.catch_warnings():
        # Ignoring warnings in this test as warnings are
        # being caught and validated in other tests.
        warnings.filterwarnings(
            "ignore",
            "The behavior of array concatenation with empty entries "
            "is deprecated.",
            category=FutureWarning,
        )
        yield


def make_frames(index=None, nulls="none"):
    rng = np.random.default_rng(seed=0)
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
        rng.shuffle(mask)
        mask = mask[:5]
        df.loc[mask, "y"] = np.nan
        df2.loc[mask, "y"] = np.nan
    gdf = cudf.DataFrame.from_pandas(df)
    gdf2 = cudf.DataFrame.from_pandas(df2)
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
    with _hide_concat_empty_dtype_warning():
        res = cudf.concat([gdf, gdf2, gdf, gdf_empty1], axis=axis).to_pandas()
        sol = pd.concat([df, df2, df, df_empty1], axis=axis)
    assert_eq(
        res,
        sol,
        check_names=False,
        check_categorical=False,
        check_index_type=True,
    )

    # Series
    for c in [i for i in ("x", "y", "z") if i != index]:
        res = cudf.concat([gdf[c], gdf2[c], gdf[c]], axis=axis).to_pandas()
        sol = pd.concat([df[c], df2[c], df[c]], axis=axis)
        assert_eq(
            res,
            sol,
            check_names=False,
            check_categorical=False,
            check_index_type=True,
        )

    # Index
    res = cudf.concat([gdf.index, gdf2.index], axis=axis).to_pandas()
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

    ga = cudf.Series(values)
    gb = cudf.Series([None])
    gs = cudf.concat([ga, gb])

    assert_eq(
        ps,
        gs,
        check_dtype=False,
        check_categorical=False,
        check_index_type=True,
    )


def test_concat_errors():
    df, df2, gdf, gdf2 = make_frames()

    # No objs
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([], {"objs": []}),
        rfunc_args_and_kwargs=([], {"objs": []}),
    )

    # All None
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([], {"objs": [None, None]}),
        rfunc_args_and_kwargs=([], {"objs": [None, None]}),
    )

    # Mismatched types
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([], {"objs": [df, df.index, df.x]}),
        rfunc_args_and_kwargs=([], {"objs": [gdf, gdf.index, gdf.x]}),
    )

    # Unknown type
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([], {"objs": ["bar", "foo"]}),
        rfunc_args_and_kwargs=([], {"objs": ["bar", "foo"]}),
    )

    # Mismatched index dtypes
    gdf3 = gdf2.copy()
    del gdf3["z"]
    gdf4 = gdf2.set_index("z")

    with pytest.raises(ValueError, match="All columns must be the same type"):
        cudf.concat([gdf3, gdf4])

    # Bad axis value
    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=(
            [],
            {"objs": [gdf.to_pandas(), gdf2.to_pandas()], "axis": "bad_value"},
        ),
        rfunc_args_and_kwargs=([], {"objs": [gdf, gdf2], "axis": "bad_value"}),
    )


def test_concat_misordered_columns():
    df, df2, gdf, gdf2 = make_frames(False)
    gdf2 = gdf2[["z", "x", "y"]]
    df2 = df2[["z", "x", "y"]]

    res = cudf.concat([gdf, gdf2]).to_pandas()
    sol = pd.concat([df, df2], sort=False)

    assert_eq(
        res,
        sol,
        check_names=False,
        check_categorical=False,
        check_index_type=True,
    )


@pytest.mark.parametrize("axis", [1, "columns"])
def test_concat_columns(axis):
    rng = np.random.default_rng(seed=0)
    pdf1 = pd.DataFrame(rng.integers(10, size=(5, 3)), columns=[1, 2, 3])
    pdf2 = pd.DataFrame(rng.integers(10, size=(5, 4)), columns=[4, 5, 6, 7])
    gdf1 = cudf.from_pandas(pdf1)
    gdf2 = cudf.from_pandas(pdf2)

    expect = pd.concat([pdf1, pdf2], axis=axis)
    got = cudf.concat([gdf1, gdf2], axis=axis)

    assert_eq(expect, got, check_index_type=True)


@pytest.mark.parametrize("axis", [0, 1])
def test_concat_multiindex_dataframe(axis):
    gdf = cudf.DataFrame(
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
    gdg1 = cudf.from_pandas(pdg1)
    gdg2 = cudf.from_pandas(pdg2)
    expected = pd.concat([pdg1, pdg2], axis=axis)
    result = cudf.concat([gdg1, gdg2], axis=axis)
    assert_eq(
        expected,
        result,
        check_index_type=True,
    )


def test_concat_multiindex_series():
    gdf = cudf.DataFrame(
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
    gdg1 = cudf.from_pandas(pdg1)
    gdg2 = cudf.from_pandas(pdg2)
    assert_eq(
        cudf.concat([gdg1, gdg2]),
        pd.concat([pdg1, pdg2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([gdg1, gdg2], axis=1), pd.concat([pdg1, pdg2], axis=1)
    )


def test_concat_multiindex_dataframe_and_series():
    gdf = cudf.DataFrame(
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
    gdg1 = cudf.from_pandas(pdg1)
    gdg2 = cudf.from_pandas(pdg2)
    assert_eq(
        cudf.concat([gdg1, gdg2], axis=1),
        pd.concat([pdg1, pdg2], axis=1),
        check_index_type=True,
    )


def test_concat_multiindex_series_and_dataframe():
    gdf = cudf.DataFrame(
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
    gdg1 = cudf.from_pandas(pdg1)
    gdg2 = cudf.from_pandas(pdg2)
    assert_eq(
        cudf.concat([gdg1, gdg2], axis=1),
        pd.concat([pdg1, pdg2], axis=1),
        check_index_type=True,
    )


@pytest.mark.parametrize("myindex", ["a", "b"])
def test_concat_string_index_name(myindex):
    # GH-Issue #3420
    data = {"a": [123, 456], "b": ["s1", "s2"]}
    df1 = cudf.DataFrame(data).set_index(myindex)
    df2 = df1.copy()
    df3 = cudf.concat([df1, df2])

    assert df3.index.name == myindex


def test_pandas_concat_compatibility_axis1():
    d1 = cudf.datasets.randomdata(
        3, dtypes={"a": float, "ind": float}
    ).set_index("ind")
    d2 = cudf.datasets.randomdata(
        3, dtypes={"b": float, "ind": float}
    ).set_index("ind")
    d3 = cudf.datasets.randomdata(
        3, dtypes={"c": float, "ind": float}
    ).set_index("ind")
    d4 = cudf.datasets.randomdata(
        3, dtypes={"d": float, "ind": float}
    ).set_index("ind")
    d5 = cudf.datasets.randomdata(
        3, dtypes={"e": float, "ind": float}
    ).set_index("ind")

    pd1 = d1.to_pandas()
    pd2 = d2.to_pandas()
    pd3 = d3.to_pandas()
    pd4 = d4.to_pandas()
    pd5 = d5.to_pandas()

    expect = pd.concat([pd1, pd2, pd3, pd4, pd5], axis=1)
    got = cudf.concat([d1, d2, d3, d4, d5], axis=1)

    assert_eq(
        got.sort_index(),
        expect.sort_index(),
        check_index_type=True,
    )


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
    s1 = cudf.Series(data[0], index=[0, 1, 2])
    s2 = cudf.Series(data[1], index=index)
    if names:
        s1.name = names[0]
        s2.name = names[1]
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    got = cudf.concat([s1, s2], axis=1)
    expect = pd.concat([ps1, ps2], axis=1)
    assert_eq(got, expect, check_index_type=True)


def test_pandas_concat_compatibility_axis1_eq_index():
    s1 = cudf.Series(["a", "b", "c"], index=[0, 1, 2])
    s2 = cudf.Series(["a", "b", "c"], index=[1, 1, 1])
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()

    assert_exceptions_equal(
        lfunc=pd.concat,
        rfunc=cudf.concat,
        lfunc_args_and_kwargs=([], {"objs": [ps1, ps2], "axis": 1}),
        rfunc_args_and_kwargs=([], {"objs": [s1, s2], "axis": 1}),
    )


@pytest.mark.parametrize("name", [None, "a"])
def test_pandas_concat_compatibility_axis1_single_column(name):
    # Pandas renames series name `None` to 0
    # and preserves anything else
    s = cudf.Series([1, 2, 3], name=name)
    got = cudf.concat([s], axis=1)
    expected = pd.concat([s.to_pandas()], axis=1)
    assert_eq(expected, got)


def test_concat_duplicate_columns():
    cdf = cudf.DataFrame(
        {
            "id4": 4 * list(range(6)),
            "id5": 4 * list(reversed(range(6))),
            "v3": 6 * list(range(4)),
        }
    )
    cdf_std = cdf.groupby(["id4", "id5"])[["v3"]].std()
    cdf_med = cdf.groupby(["id4", "id5"])[["v3"]].quantile(q=0.5)
    with pytest.raises(NotImplementedError):
        cudf.concat([cdf_med, cdf_std], axis=1)


def test_concat_mixed_input():
    pdf1 = pd.DataFrame({"a": [10, 20, 30]})
    pdf2 = pd.DataFrame({"a": [11, 22, 33]})

    gdf1 = cudf.from_pandas(pdf1)
    gdf2 = cudf.from_pandas(pdf2)

    assert_eq(
        pd.concat([pdf1, None, pdf2, None]),
        cudf.concat([gdf1, None, gdf2, None]),
        check_index_type=True,
    )
    assert_eq(
        pd.concat([pdf1, None]),
        cudf.concat([gdf1, None]),
        check_index_type=True,
    )
    assert_eq(
        pd.concat([None, pdf2]),
        cudf.concat([None, gdf2]),
        check_index_type=True,
    )
    assert_eq(
        pd.concat([None, pdf2, pdf1]),
        cudf.concat([None, gdf2, gdf1]),
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [pd.Series([1, 2, 3]), pd.DataFrame({"a": [1, 2]})],
        [pd.Series([1, 2, 3]), pd.DataFrame({"a": []})],
        [pd.Series([], dtype="float64"), pd.DataFrame({"a": []})],
        [pd.Series([], dtype="float64"), pd.DataFrame({"a": [1, 2]})],
        pytest.param(
            [
                pd.Series([1, 2, 3.0, 1.2], name="abc"),
                pd.DataFrame({"a": [1, 2]}),
            ],
            marks=pytest.mark.skipif(
                not PANDAS_GE_220,
                reason="https://github.com/pandas-dev/pandas/pull/56365",
            ),
        ),
        pytest.param(
            [
                pd.Series(
                    [1, 2, 3.0, 1.2], name="abc", index=[100, 110, 120, 130]
                ),
                pd.DataFrame({"a": [1, 2]}),
            ],
            marks=pytest.mark.skipif(
                not PANDAS_GE_220,
                reason="https://github.com/pandas-dev/pandas/pull/56365",
            ),
        ),
        pytest.param(
            [
                pd.Series(
                    [1, 2, 3.0, 1.2], name="abc", index=["a", "b", "c", "d"]
                ),
                pd.DataFrame({"a": [1, 2]}, index=["a", "b"]),
            ],
            marks=pytest.mark.skipif(
                not PANDAS_GE_220,
                reason="https://github.com/pandas-dev/pandas/pull/56365",
            ),
        ),
        pytest.param(
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
            marks=pytest.mark.skipif(
                not PANDAS_GE_220,
                reason="https://github.com/pandas-dev/pandas/pull/56365",
            ),
        ),
        pytest.param(
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
            marks=pytest.mark.skipif(
                not PANDAS_GE_220,
                reason="https://github.com/pandas-dev/pandas/pull/56365",
            ),
        ),
    ],
)
def test_concat_series_dataframe_input(objs):
    pd_objs = objs
    gd_objs = [cudf.from_pandas(obj) for obj in objs]

    with _hide_concat_empty_dtype_warning():
        expected = pd.concat(pd_objs)
        actual = cudf.concat(gd_objs)

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
    gd_objs = [cudf.from_pandas(obj) for obj in objs]

    expected = pd.concat(pd_objs)
    actual = cudf.concat(gd_objs)
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

    gdf = cudf.from_pandas(df)
    other_gd = [gdf] + [cudf.from_pandas(o) for o in other]

    expected = pd.concat(other_pd, ignore_index=ignore_index)
    actual = cudf.concat(other_gd, ignore_index=ignore_index)
    if expected.shape != df.shape:
        for key, col in actual[actual.columns].items():
            if isinstance(col.dtype, cudf.CategoricalDtype):
                if not isinstance(expected[key].dtype, pd.CategoricalDtype):
                    # TODO: Pandas bug:
                    # https://github.com/pandas-dev/pandas/issues/42840
                    expected[key] = expected[key].fillna("-1").astype("str")
                else:
                    expected[key] = (
                        expected[key]
                        .cat.add_categories(["-1"])
                        .fillna("-1")
                        .astype("str")
                    )
                actual[key] = col.astype("str").fillna("-1")
            else:
                expected[key] = expected[key].fillna(-1)
                actual[key] = col.fillna(-1)
        assert_eq(expected, actual, check_dtype=False, check_index_type=True)
    else:
        assert_eq(
            expected,
            actual,
            check_index_type=not gdf.empty,
            check_column_type=False,
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
    s1 = cudf.Series()
    s2 = cudf.Series(data[0])
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    got = cudf.concat([s1, s2], axis=axis, ignore_index=ignore_index)
    expect = pd.concat([ps1, ps2], axis=axis, ignore_index=ignore_index)

    assert_eq(got, expect, check_index_type=True)


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("axis", [0, "index"])
def test_concat_two_empty_series(ignore_index, axis):
    s1 = cudf.Series()
    s2 = cudf.Series()
    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    got = cudf.concat([s1, s2], axis=axis, ignore_index=ignore_index)
    expect = pd.concat([ps1, ps2], axis=axis, ignore_index=ignore_index)

    assert_eq(got, expect, check_index_type=True)


@pytest.mark.parametrize(
    "df1,df2",
    [
        (
            cudf.DataFrame({"k1": [0, 1], "k2": [2, 3], "v1": [4, 5]}),
            cudf.DataFrame({"k1": [1, 0], "k2": [3, 2], "v2": [6, 7]}),
        ),
        (
            cudf.DataFrame({"k1": [0, 1], "k2": [2, 3], "v1": [4, 5]}),
            cudf.DataFrame({"k1": [0, 1], "k2": [3, 2], "v2": [6, 7]}),
        ),
    ],
)
def test_concat_dataframe_with_multiindex(df1, df2):
    gdf1 = df1
    gdf1 = gdf1.set_index(["k1", "k2"])

    gdf2 = df2
    gdf2 = gdf2.set_index(["k1", "k2"])

    pdf1 = gdf1.to_pandas()
    pdf2 = gdf2.to_pandas()

    actual = cudf.concat([gdf1, gdf2], axis=1)
    expected = pd.concat([pdf1, pdf2], axis=1)

    # Will need to sort_index before comparing as
    # ordering is not deterministic in case of pandas
    # multiIndex with concat.
    assert_eq(
        expected.sort_index(),
        actual.sort_index(),
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }
            ),
            pd.DataFrame(
                {"x": range(10, 20), "y": list(map(float, range(10, 20)))}
            ),
        ],
        [
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            ),
            pd.DataFrame(
                {"x": range(10, 20), "y": list(map(float, range(10, 20)))},
                index=["k", "l", "m", "n", "o", "p", "q", "r", "s", "t"],
            ),
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            ),
            pd.DataFrame(
                {"x": range(10, 20), "y": list(map(float, range(10, 20)))},
                index=["a", "b", "c", "d", "z", "f", "g", "h", "i", "w"],
            ),
        ],
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("axis", [0])
def test_concat_join(objs, ignore_index, sort, join, axis):
    gpu_objs = [cudf.from_pandas(o) for o in objs]

    assert_eq(
        pd.concat(
            objs, sort=sort, join=join, ignore_index=ignore_index, axis=axis
        ),
        cudf.concat(
            gpu_objs,
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        ),
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }
            ),
            pd.DataFrame(
                {"x": range(10, 20), "y": list(map(float, range(10, 20)))}
            ),
        ],
    ],
)
def test_concat_join_axis_1_dup_error(objs):
    gpu_objs = [cudf.from_pandas(o) for o in objs]
    # we do not support duplicate columns
    with pytest.raises(NotImplementedError):
        assert_eq(
            pd.concat(
                objs,
                axis=1,
            ),
            cudf.concat(
                gpu_objs,
                axis=1,
            ),
        )


@pytest.mark.parametrize(
    "objs",
    [
        [
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }
            ),
            pd.DataFrame(
                {"l": range(10, 20), "m": list(map(float, range(10, 20)))}
            ),
        ],
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("axis", [1])
def test_concat_join_axis_1(objs, ignore_index, sort, join, axis):
    # no duplicate columns
    gpu_objs = [cudf.from_pandas(o) for o in objs]
    expected = pd.concat(
        objs, sort=sort, join=join, ignore_index=ignore_index, axis=axis
    )
    actual = cudf.concat(
        gpu_objs,
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )

    assert_eq(expected, actual, check_index_type=True)


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("axis", [1, 0])
def test_concat_join_many_df_and_empty_df(ignore_index, sort, join, axis):
    # no duplicate columns
    pdf1 = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    pdf2 = pd.DataFrame(
        {"l": range(10, 20), "m": list(map(float, range(10, 20)))}
    )
    pdf3 = pd.DataFrame({"j": [1, 2], "k": [1, 2], "s": [1, 2], "t": [1, 2]})
    pdf_empty1 = pd.DataFrame()

    gdf1 = cudf.from_pandas(pdf1)
    gdf2 = cudf.from_pandas(pdf2)
    gdf3 = cudf.from_pandas(pdf3)
    gdf_empty1 = cudf.from_pandas(pdf_empty1)

    with _hide_concat_empty_dtype_warning():
        assert_eq(
            pd.concat(
                [pdf1, pdf2, pdf3, pdf_empty1],
                sort=sort,
                join=join,
                ignore_index=ignore_index,
                axis=axis,
            ),
            cudf.concat(
                [gdf1, gdf2, gdf3, gdf_empty1],
                sort=sort,
                join=join,
                ignore_index=ignore_index,
                axis=axis,
            ),
            check_index_type=False,
        )


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("axis", [0, 1])
def test_concat_join_one_df(ignore_index, sort, join, axis):
    pdf1 = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    gdf1 = cudf.from_pandas(pdf1)
    expected = pd.concat(
        [pdf1], sort=sort, join=join, ignore_index=ignore_index, axis=axis
    )
    actual = cudf.concat(
        [gdf1], sort=sort, join=join, ignore_index=ignore_index, axis=axis
    )

    assert_eq(expected, actual, check_index_type=True)


@pytest.mark.parametrize(
    "pdf1,pdf2",
    [
        (
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]}),
        ),
        (
            pd.DataFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6]}, index=["p", "q", "r"]
            ),
            pd.DataFrame(
                {"c": [7, 8, 9], "d": [10, 11, 12]}, index=["r", "p", "z"]
            ),
        ),
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("axis", [0, 1])
def test_concat_join_no_overlapping_columns(
    pdf1, pdf2, ignore_index, sort, join, axis
):
    gdf1 = cudf.from_pandas(pdf1)
    gdf2 = cudf.from_pandas(pdf2)

    expected = pd.concat(
        [pdf1, pdf2],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )
    actual = cudf.concat(
        [gdf1, gdf2],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )

    assert_eq(expected, actual, check_index_type=True)


@pytest.mark.parametrize("ignore_index", [False, True])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("axis", [0, 1])
def test_concat_join_no_overlapping_columns_many_and_empty(
    ignore_index, sort, join, axis
):
    pdf4 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pdf5 = pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})
    pdf6 = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    pdf_empty = pd.DataFrame()

    gdf4 = cudf.from_pandas(pdf4)
    gdf5 = cudf.from_pandas(pdf5)
    gdf6 = cudf.from_pandas(pdf6)
    gdf_empty = cudf.from_pandas(pdf_empty)

    with _hide_concat_empty_dtype_warning():
        expected = pd.concat(
            [pdf4, pdf5, pdf6, pdf_empty],
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        )
        actual = cudf.concat(
            [gdf4, gdf5, gdf6, gdf_empty],
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        )
    assert_eq(
        expected,
        actual,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    "objs",
    [
        [
            pd.DataFrame(
                {"a": [1, 2, 3], "b": [4, 5, 6]}, index=["z", "t", "k"]
            ),
            pd.DataFrame(
                {"c": [7, 8, 9], "d": [10, 11, 12]}, index=["z", "t", "k"]
            ),
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                index=["z", "t", "k", "a", "b", "c", "d", "e", "f", "g"],
            ),
            pd.DataFrame(index=pd.Index([], dtype="str")),
        ],
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]}),
            pd.DataFrame(
                {
                    "x": range(10),
                    "y": list(map(float, range(10))),
                    "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }
            ),
            pd.DataFrame(index=pd.Index([], dtype="str")),
        ],
        pytest.param(
            [
                pd.DataFrame(
                    {"a": [1, 2, 3], "nb": [10, 11, 12]}, index=["Q", "W", "R"]
                ),
                None,
            ],
        ),
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("join", ["outer", "inner"])
@pytest.mark.parametrize("axis", [0, 1])
def test_concat_join_no_overlapping_columns_many_and_empty2(
    objs, ignore_index, sort, join, axis
):
    objs_gd = [cudf.from_pandas(o) if o is not None else o for o in objs]

    with _hide_concat_empty_dtype_warning():
        expected = pd.concat(
            objs,
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        )
        actual = cudf.concat(
            objs_gd,
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        )
    assert_eq(expected, actual, check_index_type=False)


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("axis", [0, 1])
def test_concat_join_no_overlapping_columns_empty_df_basic(
    ignore_index, sort, join, axis
):
    pdf6 = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    pdf_empty = pd.DataFrame()

    gdf6 = cudf.from_pandas(pdf6)
    gdf_empty = cudf.from_pandas(pdf_empty)

    with _hide_concat_empty_dtype_warning():
        expected = pd.concat(
            [pdf6, pdf_empty],
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        )
        actual = cudf.concat(
            [gdf6, gdf_empty],
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        )
    assert_eq(
        expected,
        actual,
        check_index_type=True,
        check_column_type=False,
    )


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("axis", [0, 1])
def test_concat_join_series(ignore_index, sort, join, axis):
    s1 = cudf.Series(["a", "b", "c"])
    s2 = cudf.Series(["a", "b"])
    s3 = cudf.Series(["a", "b", "c", "d"])
    s4 = cudf.Series(dtype="str")

    ps1 = s1.to_pandas()
    ps2 = s2.to_pandas()
    ps3 = s3.to_pandas()
    ps4 = s4.to_pandas()

    expected = pd.concat(
        [ps1, ps2, ps3, ps4],
        sort=sort,
        join=join,
        ignore_index=ignore_index,
        axis=axis,
    )
    with expect_warning_if(axis == 1):
        actual = cudf.concat(
            [s1, s2, s3, s4],
            sort=sort,
            join=join,
            ignore_index=ignore_index,
            axis=axis,
        )

    assert_eq(
        expected,
        actual,
        check_index_type=True,
    )


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
                {"b": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        ],
        [
            pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
            pd.DataFrame({"l": [10]}),
            pd.DataFrame({"k": [10]}, index=[200]),
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
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
def test_concat_join_empty_dataframes(
    request, df, other, ignore_index, join, sort
):
    axis = 0
    other_pd = [df] + other
    gdf = cudf.from_pandas(df)
    other_gd = [gdf] + [cudf.from_pandas(o) for o in other]

    expected = pd.concat(
        other_pd, ignore_index=ignore_index, axis=axis, join=join, sort=sort
    )
    actual = cudf.concat(
        other_gd, ignore_index=ignore_index, axis=axis, join=join, sort=sort
    )
    if (
        join == "outer"
        and any(
            isinstance(dtype, pd.CategoricalDtype)
            for dtype in df.dtypes.tolist()
        )
        and any(
            isinstance(dtype, pd.CategoricalDtype)
            for other_df in other
            for dtype in other_df.dtypes.tolist()
        )
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/42840"
            )
        )
    assert_eq(
        expected,
        actual,
        check_dtype=False,
        check_column_type=False,
    )


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
        pd.DataFrame({"m": [10]}, index=[200]),
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
                {"b": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("CD")),
        ],
        [
            pd.DataFrame({"g": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
            pd.DataFrame({"h": [10]}),
            pd.DataFrame({"k": [10]}, index=[200]),
            pd.DataFrame(
                {"dog": pd.Series(["two", "three"], dtype="category")}
            ),
        ],
        [
            pd.DataFrame([]),
            pd.DataFrame([], index=[100]),
            pd.DataFrame(
                {"bird": pd.Series(["two", "three"], dtype="category")}
            ),
        ],
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("axis", [1])
def test_concat_join_empty_dataframes_axis_1(
    df, other, ignore_index, axis, join, sort
):
    # no duplicate columns
    other_pd = [df] + other
    gdf = cudf.from_pandas(df)
    other_gd = [gdf] + [cudf.from_pandas(o) for o in other]

    with _hide_concat_empty_dtype_warning():
        expected = pd.concat(
            other_pd,
            ignore_index=ignore_index,
            axis=axis,
            join=join,
            sort=sort,
        )
        actual = cudf.concat(
            other_gd,
            ignore_index=ignore_index,
            axis=axis,
            join=join,
            sort=sort,
        )
    if expected.shape != df.shape:
        if axis == 0:
            for key, col in actual[actual.columns].items():
                if isinstance(expected[key].dtype, pd.CategoricalDtype):
                    expected[key] = expected[key].fillna("-1")
                    actual[key] = col.astype("str").fillna("-1")
            # if not expected.empty:
            assert_eq(
                expected.fillna(-1),
                actual.fillna(-1),
                check_dtype=False,
                check_index_type=False
                if len(expected) == 0 or actual.empty
                else True,
                check_column_type=False,
            )
        else:
            # no need to fill in if axis=1
            assert_eq(
                expected,
                actual,
                check_index_type=False,
                check_column_type=False,
            )
    assert_eq(
        expected, actual, check_index_type=False, check_column_type=False
    )


def test_concat_preserve_order():
    """Ensure that order is preserved on 'inner' concatenations."""
    df = pd.DataFrame([["d", 3, 4.0], ["c", 4, 5.0]], columns=["c", "b", "a"])
    dfs = [df, df]

    assert_eq(
        pd.concat(dfs, join="inner"),
        cudf.concat([cudf.DataFrame(df) for df in dfs], join="inner"),
        check_index_type=True,
    )


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("typ", [cudf.DataFrame, cudf.Series])
def test_concat_single_object(ignore_index, typ):
    """Ensure that concat on a single object does not change it."""
    obj = typ([1, 2, 3])
    assert_eq(
        cudf.concat([obj], ignore_index=ignore_index, axis=0),
        obj,
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "ltype",
    [Decimal64Dtype(3, 1), Decimal64Dtype(7, 2), Decimal64Dtype(8, 4)],
)
@pytest.mark.parametrize(
    "rtype",
    [
        Decimal64Dtype(3, 2),
        Decimal64Dtype(8, 4),
        cudf.Decimal128Dtype(3, 2),
        cudf.Decimal32Dtype(8, 4),
    ],
)
def test_concat_decimal_dataframe(ltype, rtype):
    rng = np.random.default_rng(seed=0)
    gdf1 = cudf.DataFrame(
        {"id": rng.integers(0, 10, 3), "val": ["22.3", "59.5", "81.1"]}
    )
    gdf2 = cudf.DataFrame(
        {"id": rng.integers(0, 10, 3), "val": ["2.35", "5.59", "8.14"]}
    )

    gdf1["val"] = gdf1["val"].astype(ltype)
    gdf2["val"] = gdf2["val"].astype(rtype)

    pdf1 = gdf1.to_pandas()
    pdf2 = gdf2.to_pandas()

    got = cudf.concat([gdf1, gdf2])
    expected = pd.concat([pdf1, pdf2])

    assert_eq(expected, got, check_index_type=True)


@pytest.mark.parametrize("ltype", [Decimal64Dtype(4, 1), Decimal64Dtype(8, 2)])
@pytest.mark.parametrize(
    "rtype",
    [
        Decimal64Dtype(4, 3),
        Decimal64Dtype(10, 4),
        Decimal32Dtype(8, 3),
        Decimal128Dtype(18, 3),
    ],
)
def test_concat_decimal_series(ltype, rtype):
    gs1 = cudf.Series(["228.3", "559.5", "281.1"]).astype(ltype)
    gs2 = cudf.Series(["2.345", "5.259", "8.154"]).astype(rtype)

    ps1 = gs1.to_pandas()
    ps2 = gs2.to_pandas()

    got = cudf.concat([gs1, gs2])
    expected = pd.concat([ps1, ps2])

    assert_eq(expected, got, check_index_type=True)


@pytest.mark.parametrize(
    "df1, df2, df3, expected",
    [
        (
            cudf.DataFrame(
                {"val": [Decimal("42.5"), Decimal("8.7")]},
                dtype=Decimal64Dtype(5, 2),
            ),
            cudf.DataFrame(
                {"val": [Decimal("9.23"), Decimal("-67.49")]},
                dtype=Decimal64Dtype(6, 4),
            ),
            cudf.DataFrame({"val": [8, -5]}, dtype="int32"),
            cudf.DataFrame(
                {
                    "val": [
                        Decimal("42.5"),
                        Decimal("8.7"),
                        Decimal("9.23"),
                        Decimal("-67.49"),
                        Decimal("8"),
                        Decimal("-5"),
                    ]
                },
                dtype=Decimal32Dtype(7, 4),
                index=[0, 1, 0, 1, 0, 1],
            ),
        ),
        (
            cudf.DataFrame(
                {"val": [Decimal("95.2"), Decimal("23.4")]},
                dtype=Decimal64Dtype(5, 2),
            ),
            cudf.DataFrame({"val": [54, 509]}, dtype="uint16"),
            cudf.DataFrame({"val": [24, -48]}, dtype="int32"),
            cudf.DataFrame(
                {
                    "val": [
                        Decimal("95.2"),
                        Decimal("23.4"),
                        Decimal("54"),
                        Decimal("509"),
                        Decimal("24"),
                        Decimal("-48"),
                    ]
                },
                dtype=Decimal32Dtype(5, 2),
                index=[0, 1, 0, 1, 0, 1],
            ),
        ),
        (
            cudf.DataFrame(
                {"val": [Decimal("36.56"), Decimal("-59.24")]},
                dtype=Decimal64Dtype(9, 4),
            ),
            cudf.DataFrame({"val": [403.21, 45.13]}, dtype="float32"),
            cudf.DataFrame({"val": [52.262, -49.25]}, dtype="float64"),
            cudf.DataFrame(
                {
                    "val": [
                        Decimal("36.56"),
                        Decimal("-59.24"),
                        Decimal("403.21"),
                        Decimal("45.13"),
                        Decimal("52.262"),
                        Decimal("-49.25"),
                    ]
                },
                dtype=Decimal32Dtype(9, 4),
                index=[0, 1, 0, 1, 0, 1],
            ),
        ),
        (
            cudf.DataFrame(
                {"val": [Decimal("9563.24"), Decimal("236.633")]},
                dtype=Decimal64Dtype(9, 4),
            ),
            cudf.DataFrame({"val": [5393, -95832]}, dtype="int64"),
            cudf.DataFrame({"val": [-29.234, -31.945]}, dtype="float64"),
            cudf.DataFrame(
                {
                    "val": [
                        Decimal("9563.24"),
                        Decimal("236.633"),
                        Decimal("5393"),
                        Decimal("-95832"),
                        Decimal("-29.234"),
                        Decimal("-31.945"),
                    ]
                },
                dtype=Decimal32Dtype(9, 4),
                index=[0, 1, 0, 1, 0, 1],
            ),
        ),
        (
            cudf.DataFrame(
                {"val": [Decimal("95633.24"), Decimal("236.633")]},
                dtype=Decimal128Dtype(19, 4),
            ),
            cudf.DataFrame({"val": [5393, -95832]}, dtype="int64"),
            cudf.DataFrame({"val": [-29.234, -31.945]}, dtype="float64"),
            cudf.DataFrame(
                {
                    "val": [
                        Decimal("95633.24"),
                        Decimal("236.633"),
                        Decimal("5393"),
                        Decimal("-95832"),
                        Decimal("-29.234"),
                        Decimal("-31.945"),
                    ]
                },
                dtype=Decimal128Dtype(19, 4),
                index=[0, 1, 0, 1, 0, 1],
            ),
        ),
    ],
)
def test_concat_decimal_numeric_dataframe(df1, df2, df3, expected):
    df = cudf.concat([df1, df2, df3])
    assert_eq(df, expected, check_index_type=True)
    assert_eq(df.val.dtype, expected.val.dtype)


@pytest.mark.parametrize(
    "s1, s2, s3, expected",
    [
        (
            cudf.Series(
                [Decimal("32.8"), Decimal("-87.7")], dtype=Decimal64Dtype(6, 2)
            ),
            cudf.Series(
                [Decimal("101.243"), Decimal("-92.449")],
                dtype=Decimal64Dtype(9, 6),
            ),
            cudf.Series([94, -22], dtype="int32"),
            cudf.Series(
                [
                    Decimal("32.8"),
                    Decimal("-87.7"),
                    Decimal("101.243"),
                    Decimal("-92.449"),
                    Decimal("94"),
                    Decimal("-22"),
                ],
                dtype=Decimal64Dtype(10, 6),
                index=[0, 1, 0, 1, 0, 1],
            ),
        ),
        (
            cudf.Series(
                [Decimal("7.2"), Decimal("122.1")], dtype=Decimal64Dtype(5, 2)
            ),
            cudf.Series([33, 984], dtype="uint32"),
            cudf.Series([593, -702], dtype="int32"),
            cudf.Series(
                [
                    Decimal("7.2"),
                    Decimal("122.1"),
                    Decimal("33"),
                    Decimal("984"),
                    Decimal("593"),
                    Decimal("-702"),
                ],
                dtype=Decimal32Dtype(5, 2),
                index=[0, 1, 0, 1, 0, 1],
            ),
        ),
        (
            cudf.Series(
                [Decimal("982.94"), Decimal("-493.626")],
                dtype=Decimal64Dtype(9, 4),
            ),
            cudf.Series([847.98, 254.442], dtype="float32"),
            cudf.Series([5299.262, -2049.25], dtype="float64"),
            cudf.Series(
                [
                    Decimal("982.94"),
                    Decimal("-493.626"),
                    Decimal("847.98"),
                    Decimal("254.442"),
                    Decimal("5299.262"),
                    Decimal("-2049.25"),
                ],
                dtype=Decimal32Dtype(9, 4),
                index=[0, 1, 0, 1, 0, 1],
            ),
        ),
        (
            cudf.Series(
                [Decimal("492.204"), Decimal("-72824.455")],
                dtype=Decimal64Dtype(9, 4),
            ),
            cudf.Series([8438, -27462], dtype="int64"),
            cudf.Series([-40.292, 49202.953], dtype="float64"),
            cudf.Series(
                [
                    Decimal("492.204"),
                    Decimal("-72824.455"),
                    Decimal("8438"),
                    Decimal("-27462"),
                    Decimal("-40.292"),
                    Decimal("49202.953"),
                ],
                dtype=Decimal32Dtype(9, 4),
                index=[0, 1, 0, 1, 0, 1],
            ),
        ),
        (
            cudf.Series(
                [Decimal("492.204"), Decimal("-72824.455")],
                dtype=Decimal64Dtype(10, 4),
            ),
            cudf.Series(
                [Decimal("8438"), Decimal("-27462")],
                dtype=Decimal32Dtype(9, 4),
            ),
            cudf.Series(
                [Decimal("-40.292"), Decimal("49202.953")],
                dtype=Decimal128Dtype(19, 4),
            ),
            cudf.Series(
                [
                    Decimal("492.204"),
                    Decimal("-72824.455"),
                    Decimal("8438"),
                    Decimal("-27462"),
                    Decimal("-40.292"),
                    Decimal("49202.953"),
                ],
                dtype=Decimal128Dtype(19, 4),
                index=[0, 1, 0, 1, 0, 1],
            ),
        ),
    ],
)
def test_concat_decimal_numeric_series(s1, s2, s3, expected):
    s = cudf.concat([s1, s2, s3])
    assert_eq(s, expected, check_index_type=True)


@pytest.mark.parametrize(
    "s1, s2, expected",
    [
        (
            cudf.Series(
                [Decimal("955.22"), Decimal("8.2")], dtype=Decimal64Dtype(5, 2)
            ),
            cudf.Series(["2007-06-12", "2006-03-14"], dtype="datetime64[s]"),
            cudf.Series(
                [
                    "955.22",
                    "8.20",
                    "2007-06-12 00:00:00",
                    "2006-03-14 00:00:00",
                ],
                index=[0, 1, 0, 1],
            ),
        ),
        (
            cudf.Series(
                [Decimal("-52.44"), Decimal("365.22")],
                dtype=Decimal64Dtype(5, 2),
            ),
            cudf.Series(
                np.arange(
                    "2005-02-01T12", "2005-02-01T15", dtype="datetime64[h]"
                ).astype("datetime64[s]"),
                dtype="datetime64[s]",
            ),
            cudf.Series(
                [
                    "-52.44",
                    "365.22",
                    "2005-02-01 12:00:00",
                    "2005-02-01 13:00:00",
                    "2005-02-01 14:00:00",
                ],
                index=[0, 1, 0, 1, 2],
            ),
        ),
        (
            cudf.Series(
                [Decimal("753.0"), Decimal("94.22")],
                dtype=Decimal64Dtype(5, 2),
            ),
            cudf.Series([np.timedelta64(111, "s"), np.timedelta64(509, "s")]),
            cudf.Series(
                ["753.00", "94.22", "0 days 00:01:51", "0 days 00:08:29"],
                index=[0, 1, 0, 1],
            ),
        ),
        (
            cudf.Series(
                [Decimal("753.0"), Decimal("94.22")],
                dtype=Decimal64Dtype(5, 2),
            ),
            cudf.Series(
                [np.timedelta64(940252, "s"), np.timedelta64(758385, "s")]
            ),
            cudf.Series(
                ["753.00", "94.22", "10 days 21:10:52", "8 days 18:39:45"],
                index=[0, 1, 0, 1],
            ),
        ),
    ],
)
def test_concat_decimal_non_numeric(s1, s2, expected):
    s = cudf.concat([s1, s2])
    assert_eq(s, expected, check_index_type=True)


@pytest.mark.parametrize(
    "s1, s2, expected",
    [
        (
            cudf.Series([{"a": 5}, {"c": "hello"}, {"b": 7}]),
            cudf.Series([{"a": 5, "c": "hello", "b": 7}]),
            cudf.Series(
                [
                    {"a": 5, "b": None, "c": None},
                    {"a": None, "b": None, "c": "hello"},
                    {"a": None, "b": 7, "c": None},
                    {"a": 5, "b": 7, "c": "hello"},
                ],
                index=[0, 1, 2, 0],
            ),
        )
    ],
)
def test_concat_struct_column(s1, s2, expected):
    s = cudf.concat([s1, s2])
    assert_eq(s, expected, check_index_type=True)


@pytest.mark.parametrize(
    "frame1, frame2, expected",
    [
        (
            cudf.Series([[{"b": 0}], [{"b": 1}], [{"b": 3}]]),
            cudf.Series([[{"b": 10}], [{"b": 12}], None]),
            cudf.Series(
                [
                    [{"b": 0}],
                    [{"b": 1}],
                    [{"b": 3}],
                    [{"b": 10}],
                    [{"b": 12}],
                    None,
                ],
                index=[0, 1, 2, 0, 1, 2],
            ),
        ),
        (
            cudf.DataFrame({"a": [[{"b": 0}], [{"b": 1}], [{"b": 3}]]}),
            cudf.DataFrame({"a": [[{"b": 10}], [{"b": 12}], None]}),
            cudf.DataFrame(
                {
                    "a": [
                        [{"b": 0}],
                        [{"b": 1}],
                        [{"b": 3}],
                        [{"b": 10}],
                        [{"b": 12}],
                        None,
                    ]
                },
                index=[0, 1, 2, 0, 1, 2],
            ),
        ),
    ],
)
def test_concat_list_column(frame1, frame2, expected):
    actual = cudf.concat([frame1, frame2])
    assert_eq(actual, expected, check_index_type=True)


def test_concat_categorical_ordering():
    # https://github.com/rapidsai/cudf/issues/11486
    sr = pd.Series(
        ["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"], dtype="category"
    )
    sr = sr.cat.set_categories(["d", "a", "b", "c", "e"])

    df = pd.DataFrame({"a": sr})
    gdf = cudf.from_pandas(df)

    expect = pd.concat([df, df, df])
    got = cudf.concat([gdf, gdf, gdf])

    assert_eq(expect, got)


@pytest.fixture(params=["rangeindex", "index"])
def singleton_concat_index(request):
    if request.param == "rangeindex":
        return pd.RangeIndex(0, 4)
    else:
        return pd.Index(["a", "h", "g", "f"])


@pytest.fixture(params=["dataframe", "series"])
def singleton_concat_obj(request, singleton_concat_index):
    if request.param == "dataframe":
        return pd.DataFrame(
            {
                "b": [1, 2, 3, 4],
                "d": [7, 8, 9, 10],
                "a": [4, 5, 6, 7],
                "c": [10, 11, 12, 13],
            },
            index=singleton_concat_index,
        )
    else:
        return pd.Series([4, 5, 5, 6], index=singleton_concat_index)


@pytest.mark.parametrize("axis", [0, 1, "columns", "index"])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("ignore_index", [False, True])
def test_concat_singleton_sorting(
    axis, sort, ignore_index, singleton_concat_obj
):
    gobj = cudf.from_pandas(singleton_concat_obj)
    gconcat = cudf.concat(
        [gobj], axis=axis, sort=sort, ignore_index=ignore_index
    )
    pconcat = pd.concat(
        [singleton_concat_obj], axis=axis, sort=sort, ignore_index=ignore_index
    )
    assert_eq(pconcat, gconcat)


@pytest.mark.parametrize("axis", [2, "invalid"])
def test_concat_invalid_axis(axis):
    s = cudf.Series([1, 2, 3])
    with pytest.raises(ValueError):
        cudf.concat([s], axis=axis)


@pytest.mark.parametrize(
    "s1,s2",
    [
        ([1, 2], [[1, 2], [3, 4]]),
    ],
)
def test_concat_mixed_list_types_error(s1, s2):
    s1, s2 = cudf.Series(s1), cudf.Series(s2)

    with pytest.raises(NotImplementedError):
        cudf.concat([s1, s2], ignore_index=True)


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(
            0,
            marks=pytest.mark.xfail(
                reason="concat dictionaries with axis=0 not implemented"
            ),
        ),
        1,
        "columns",
    ],
)
@pytest.mark.parametrize(
    "d",
    [
        {"first": (cudf.DataFrame, {"data": {"A": [1, 2], "B": [3, 4]}})},
        {
            "first": (cudf.DataFrame, {"data": {"A": [1, 2], "B": [3, 4]}}),
            "second": (cudf.DataFrame, {"data": {"A": [5, 6], "B": [7, 8]}}),
            "third": (cudf.DataFrame, {"data": {"C": [1, 2, 3]}}),
        },
        {
            "first": (cudf.DataFrame, {"data": {"A": [1, 2], "B": [3, 4]}}),
            "second": (cudf.DataFrame, {"data": {"C": [1, 2, 3]}}),
            "third": (cudf.DataFrame, {"data": {"A": [5, 6], "B": [7, 8]}}),
        },
        {
            "first": (cudf.DataFrame, {"data": {"A": [1, 2], "B": [3, 4]}}),
            "second": (cudf.DataFrame, {"data": {"C": [1, 2, 3]}}),
            "third": (cudf.DataFrame, {"data": {"A": [5, 6], "C": [7, 8]}}),
            "fourth": (cudf.DataFrame, {"data": {"B": [9, 10]}}),
        },
        pytest.param(
            {
                "first": (cudf.DataFrame, {"data": {2.0: [1, 1]}}),
                "second": (cudf.DataFrame, {"data": {"test": ["abc", "def"]}}),
            },
            marks=pytest.mark.xfail(
                reason=(
                    "Cannot construct a MultiIndex column with multiple "
                    "label types in cuDF at this time. You must convert "
                    "the labels to the same type."
                )
            ),
        ),
        {
            "first": (cudf.Series, {"data": [1, 2, 3]}),
            "second": (cudf.Series, {"data": [4, 5, 6]}),
        },
        {
            "first": (cudf.DataFrame, {"data": {"A": [1, 2], "B": [3, 4]}}),
            "second": (cudf.Series, {"data": [5, 6], "name": "C"}),
        },
        pytest.param(
            {
                "first": (
                    cudf.DataFrame,
                    {"data": {("A", "B"): [1, 2], "C": [3, 4]}},
                ),
                "second": (
                    cudf.DataFrame,
                    {"data": {"D": [5, 6], ("A", "B"): [7, 8]}},
                ),
            },
            marks=pytest.mark.xfail(
                reason=(
                    "Cannot construct a MultiIndex column with multiple "
                    "label types in cuDF at this time. You must convert "
                    "the labels to the same type."
                )
            ),
        ),
        pytest.param(
            {
                "first": (
                    cudf.DataFrame,
                    {"data": {("A", "B"): [3, 4], 2.0: [1, 1]}},
                ),
                "second": (
                    cudf.DataFrame,
                    {"data": {("C", "D"): [3, 4], 3.0: [5, 6]}},
                ),
            },
            marks=pytest.mark.xfail(
                reason=(
                    "Cannot construct a MultiIndex column with multiple "
                    "label types in cuDF at this time. You must convert "
                    "the labels to the same type."
                )
            ),
        ),
        {
            "first": (
                cudf.DataFrame,
                {"data": {(1, 2): [1, 2], (3, 4): [3, 4]}},
            ),
            "second": (
                cudf.DataFrame,
                {"data": {(1, 2): [5, 6], (5, 6): [7, 8]}},
            ),
        },
    ],
)
def test_concat_dictionary(d, axis):
    _dict = {k: c(**v) for k, (c, v) in d.items()}
    result = cudf.concat(_dict, axis=axis)
    expected = cudf.from_pandas(
        pd.concat({k: df.to_pandas() for k, df in _dict.items()}, axis=axis)
    )
    assert_eq(expected, result)


@pytest.mark.parametrize(
    "d",
    [
        {"first": cudf.Index([1, 2, 3])},
        {
            "first": cudf.MultiIndex(
                levels=[[1, 2], ["blue", "red"]],
                codes=[[0, 0, 1, 1], [1, 0, 1, 0]],
            )
        },
        {"first": cudf.CategoricalIndex([1, 2, 3])},
    ],
)
def test_concat_dict_incorrect_type_index(d):
    with pytest.raises(
        TypeError,
        match="cannot concatenate a dictionary containing indices",
    ):
        cudf.concat(d, axis=1)
