# Copyright (c) 2020, NVIDIA CORPORATION.
import re
from string import ascii_letters, digits

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import (
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
    assert_eq,
)


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 24, "d": 1010},
        {"a": 1},
        {1: "a", 2: "b", 24: "c", 1010: "d"},
        {1: "a"},
    ],
)
def test_series_init_dict(data):
    pandas_series = pd.Series(data)
    cudf_series = cudf.Series(data)

    assert_eq(pandas_series, cudf_series)


@pytest.mark.parametrize(
    "data",
    [
        {
            "a": [1, 2, 3],
            "b": [2, 3, 5],
            "c": [24, 12212, 22233],
            "d": [1010, 101010, 1111],
        },
        {"a": [1]},
    ],
)
def test_series_init_dict_lists(data):
    assert_eq(pd.Series(data), cudf.Series(data))


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4],
        [1.0, 12.221, 12.34, 13.324, 324.3242],
        [-10, -1111, 100, 11, 133],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [10, 11, 12, 13],
        [0.1, 0.002, 324.2332, 0.2342],
        [-10, -1111, 100, 11, 133],
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_append_basic(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = psr.append(other_ps, ignore_index=ignore_index)
    actual = gsr.append(other_gs, ignore_index=ignore_index)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [
            "abc",
            "def",
            "this is a string",
            "this is another string",
            "a",
            "b",
            "c",
        ],
        ["a"],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [
            "abc",
            "def",
            "this is a string",
            "this is another string",
            "a",
            "b",
            "c",
        ],
        ["a"],
        ["1", "2", "3", "4", "5"],
        ["+", "-", "!", "_", "="],
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_append_basic_str(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = psr.append(other_ps, ignore_index=ignore_index)
    actual = gsr.append(other_gs, ignore_index=ignore_index)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series(
            [
                "abc",
                "def",
                "this is a string",
                "this is another string",
                "a",
                "b",
                "c",
            ],
            index=[10, 20, 30, 40, 50, 60, 70],
        ),
        pd.Series(["a"], index=[2]),
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        pd.Series(
            [
                "abc",
                "def",
                "this is a   string",
                "this is another string",
                "a",
                "b",
                "c",
            ],
            index=[10, 20, 30, 40, 50, 60, 70],
        ),
        pd.Series(["a"], index=[133]),
        pd.Series(["1", "2", "3", "4", "5"], index=[-10, 22, 33, 44, 49]),
        pd.Series(["+", "-", "!", "_", "="], index=[11, 22, 33, 44, 2]),
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_append_series_with_index(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = cudf.from_pandas(others)

    expected = psr.append(other_ps, ignore_index=ignore_index)
    actual = gsr.append(other_gs, ignore_index=ignore_index)
    assert_eq(expected, actual)


def test_series_append_error_mixed_types():
    gsr = cudf.Series([1, 2, 3, 4])
    other = cudf.Series(["a", "b", "c", "d"])

    with pytest.raises(
        TypeError,
        match="cudf does not support mixed types, please type-cast "
        "both series to same dtypes.",
    ):
        gsr.append(other)

    with pytest.raises(
        TypeError,
        match="cudf does not support mixed types, please type-cast "
        "both series to same dtypes.",
    ):
        gsr.append([gsr, other, gsr, other])


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"]),
        pd.Series(
            [1.0, 12.221, 12.34, 13.324, 324.3242],
            index=[
                "float one",
                "float two",
                "float three",
                "float four",
                "float five",
            ],
        ),
        pd.Series(
            [-10, -1111, 100, 11, 133],
            index=["one", "two", "three", "four", "five"],
        ),
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [
            pd.Series([10, 11, 12, 13], index=["a", "b", "c", "d"]),
            pd.Series([12, 14, 15, 27], index=["d", "e", "z", "x"]),
        ],
        [
            pd.Series([10, 11, 12, 13], index=["a", "b", "c", "d"]),
            pd.Series([12, 14, 15, 27], index=["d", "e", "z", "x"]),
        ]
        * 25,
        [
            pd.Series(
                [0.1, 0.002, 324.2332, 0.2342], index=["-", "+", "%", "#"]
            ),
            pd.Series([12, 14, 15, 27], index=["d", "e", "z", "x"]),
        ]
        * 46,
        [
            pd.Series(
                [-10, -1111, 100, 11, 133],
                index=["aa", "vv", "bb", "dd", "ll"],
            )
        ],
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_append_list_series_with_index(data, others, ignore_index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = [cudf.from_pandas(obj) for obj in others]

    expected = psr.append(other_ps, ignore_index=ignore_index)
    actual = gsr.append(other_gs, ignore_index=ignore_index)
    assert_eq(expected, actual)


def test_series_append_existing_buffers():
    a1 = np.arange(10, dtype=np.float64)
    gs = cudf.Series(a1)

    # Add new buffer
    a2 = cudf.Series(np.arange(5))
    gs = gs.append(a2)
    assert len(gs) == 15
    np.testing.assert_equal(gs.to_array(), np.hstack([a1, a2.to_array()]))

    # Ensure appending to previous buffer
    a3 = cudf.Series(np.arange(3))
    gs = gs.append(a3)
    assert len(gs) == 18
    a4 = np.hstack([a1, a2.to_array(), a3.to_array()])
    np.testing.assert_equal(gs.to_array(), a4)

    # Appending different dtype
    a5 = cudf.Series(np.array([1, 2, 3], dtype=np.int32))
    a6 = cudf.Series(np.array([4.5, 5.5, 6.5], dtype=np.float64))
    gs = a5.append(a6)
    np.testing.assert_equal(
        gs.to_array(), np.hstack([a5.to_array(), a6.to_array()])
    )
    gs = cudf.Series(a6).append(a5)
    np.testing.assert_equal(
        gs.to_array(), np.hstack([a6.to_array(), a5.to_array()])
    )


def test_series_column_iter_error():
    gs = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(gs)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.items()

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.iteritems()

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs._column.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(gs._column)


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, None, 4.0, 5.0],
        ["a", "b", "c", "d", "e"],
        ["a", "b", None, "d", "e"],
        [None, None, None, None, None],
        np.array(["1991-11-20", "2004-12-04"], dtype=np.datetime64),
        np.array(["1991-11-20", None], dtype=np.datetime64),
        np.array(
            ["1991-11-20 05:15:00", "2004-12-04 10:00:00"], dtype=np.datetime64
        ),
        np.array(["1991-11-20 05:15:00", None], dtype=np.datetime64),
    ],
)
def test_series_tolist(data):
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)

    with pytest.raises(
        TypeError,
        match=re.escape(
            r"cuDF does not support conversion to host memory "
            r"via `tolist()` method. Consider using "
            r"`.to_arrow().to_pylist()` to construct a Python list."
        ),
    ):
        gsr.tolist()


@pytest.mark.parametrize(
    "data",
    [[], [None, None], ["a"], ["a", "b", "c"] * 500, [1.0, 2.0, 0.3] * 57],
)
def test_series_size(data):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    assert_eq(psr.size, gsr.size)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_series_describe_numeric(dtype):
    ps = pd.Series([0, 1, 2, 3, 1, 2, 3], dtype=dtype)
    gs = cudf.from_pandas(ps)
    actual = gs.describe()
    expected = ps.describe()

    assert_eq(expected, actual)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/6219")
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
def test_series_describe_datetime(dtype):
    gs = cudf.Series([0, 1, 2, 3, 1, 2, 3], dtype=dtype)
    ps = gs.to_pandas()

    pdf_results = ps.describe(datetime_is_numeric=True)
    gdf_results = gs.describe()

    # Assert count
    p_count = pdf_results["count"]
    g_count = gdf_results["count"]

    assert_eq(int(g_count), p_count)

    # Assert Index
    assert_eq(gdf_results.index, pdf_results.index)

    # Assert rest of the element apart from
    # the first index('count')
    actual = gdf_results.tail(-1).astype("datetime64[ns]")
    expected = pdf_results.tail(-1).astype("str").astype("datetime64[ns]")

    assert_eq(expected, actual)


@pytest.mark.parametrize("dtype", TIMEDELTA_TYPES)
def test_series_describe_timedelta(dtype):
    ps = pd.Series([0, 1, 2, 3, 1, 2, 3], dtype=dtype)
    gs = cudf.from_pandas(ps)

    expected = ps.describe()
    actual = gs.describe()

    assert_eq(actual, expected.astype("str"))


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(["a", "b", "c", "d", "e", "a"]),
        pd.Series([True, False, True, True, False]),
        pd.Series([], dtype="str"),
        pd.Series(["a", "b", "c", "a"], dtype="category"),
    ],
)
def test_series_describe_other_types(ps):
    gs = cudf.from_pandas(ps)

    expected = ps.describe()
    actual = gs.describe()

    if len(ps) == 0:
        assert_eq(expected.fillna("a").astype("str"), actual.fillna("a"))
    else:
        assert_eq(expected.astype("str"), actual)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 2, 1],
        [1, 2, None, 3, 1, 1],
        [],
        ["a", "b", "c", None, "z", "a"],
    ],
)
@pytest.mark.parametrize("na_sentinel", [99999, 11, -1, 0])
def test_series_factorize(data, na_sentinel):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas()

    expected_labels, expected_cats = psr.factorize(na_sentinel=na_sentinel)
    actual_labels, actual_cats = gsr.factorize(na_sentinel=na_sentinel)

    assert_eq(expected_labels, actual_labels.to_array())
    assert_eq(expected_cats.values, actual_cats.to_array())


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_series_datetime_value_counts(data, nulls, normalize, dropna):
    psr = pd.Series(data)

    if len(data) > 0:
        if nulls == "one":
            p = np.random.randint(0, len(data))
            psr[p] = None
        elif nulls == "some":
            p = np.random.randint(0, len(data), 2)
            psr[p] = None

    gsr = cudf.from_pandas(psr)
    expected = psr.value_counts(dropna=dropna, normalize=normalize)
    got = gsr.value_counts(dropna=dropna, normalize=normalize)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=False)
    assert_eq(
        expected.reset_index(drop=True),
        got.reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("num_elements", [10, 100, 1000])
def test_categorical_value_counts(dropna, normalize, num_elements):
    # create categorical series
    np.random.seed(12)
    pd_cat = pd.Categorical(
        pd.Series(
            np.random.choice(list(ascii_letters + digits), num_elements),
            dtype="category",
        )
    )

    # gdf
    gdf = cudf.DataFrame()
    gdf["a"] = cudf.Series.from_categorical(pd_cat)
    gdf_value_counts = gdf["a"].value_counts(
        dropna=dropna, normalize=normalize
    )

    # pandas
    pdf = pd.DataFrame()
    pdf["a"] = pd_cat
    pdf_value_counts = pdf["a"].value_counts(
        dropna=dropna, normalize=normalize
    )

    # verify
    assert_eq(
        pdf_value_counts.sort_index(),
        gdf_value_counts.sort_index(),
        check_dtype=False,
    )
    assert_eq(
        pdf_value_counts.reset_index(drop=True),
        gdf_value_counts.reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_series_value_counts(dropna, normalize):
    for size in [10 ** x for x in range(5)]:
        arr = np.random.randint(low=-1, high=10, size=size)
        mask = arr != -1
        sr = cudf.Series.from_masked_array(arr, cudf.Series(mask).as_mask())
        sr.name = "col"

        expect = (
            sr.to_pandas()
            .value_counts(dropna=dropna, normalize=normalize)
            .sort_index()
        )
        got = sr.value_counts(dropna=dropna, normalize=normalize).sort_index()

        assert_eq(expect, got, check_dtype=False, check_index_type=False)


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_series_value_counts_optional_arguments(ascending, dropna, normalize):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, None])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(
        ascending=ascending, dropna=dropna, normalize=normalize
    )
    got = gsr.value_counts(
        ascending=ascending, dropna=dropna, normalize=normalize
    )

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=False)
    assert_eq(
        expected.reset_index(drop=True),
        got.reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "df",
    [
        cudf.Series([1, 2, 3]),
        cudf.Series([None]),
        cudf.Series([4]),
        cudf.Series([2, 3, -1, 0, 1], name="test name"),
        cudf.Series(
            [1, 2, 3, None, 2, 1], index=["a", "v", "d", "e", "f", "g"]
        ),
        cudf.Series([1, 2, 3, None, 2, 1, None], name="abc"),
        cudf.Series(["ab", "bc", "ab", None, "bc", None, None]),
        cudf.Series([None, None, None, None, None], dtype="str"),
        cudf.Series([None, None, None, None, None]),
        cudf.Series(
            [
                123213,
                23123,
                123123,
                12213123,
                12213123,
                12213123,
                23123,
                2312323123,
                None,
                None,
            ],
            dtype="timedelta64[ns]",
        ),
        cudf.Series(
            [
                None,
                1,
                2,
                3242434,
                3233243,
                1,
                2,
                1023,
                None,
                12213123,
                None,
                2312323123,
                None,
                None,
            ],
            dtype="datetime64[ns]",
        ),
        cudf.Series(name="empty series"),
        cudf.Series(["a", "b", "c", " ", "a", "b", "z"], dtype="category"),
    ],
)
@pytest.mark.parametrize("dropna", [True, False])
def test_series_mode(df, dropna):
    pdf = df.to_pandas()

    expected = pdf.mode(dropna=dropna)
    actual = df.mode(dropna=dropna)

    assert_eq(expected, actual, check_dtype=False)


def test_series_map():
    gd1 = cudf.Series(["cat", "dog", np.nan, "rabbit"])
    gd2 = cudf.Series([1, 2, 3, 4, np.nan])

    pdf1 = gd1.to_pandas()
    pdf2 = gd2.to_pandas()

    expected_dict = pdf1.map({"cat": "kitten", "dog": "puppy"})
    actual_dict = gd1.map({"cat": "kitten", "dog": "puppy"})

    assert_eq(expected_dict, actual_dict, check_dtype=False)

    expected_series = pdf1.map(pd.Series({"cat": "kitten", "dog": "puppy"}))
    actual_series = gd1.map(cudf.Series({"cat": "kitten", "dog": "puppy"}))

    assert_eq(expected_series, actual_series, check_dtype=False)

    expected_function = pdf2.map(lambda x: x ** 2)
    actual_function = gd2.map(lambda x: x ** 2)

    assert_eq(expected_function, actual_function, check_dtype=False)
