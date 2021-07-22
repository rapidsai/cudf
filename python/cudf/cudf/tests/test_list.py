# Copyright (c) 2020-2021, NVIDIA CORPORATION.
import functools
import operator

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf import NA
from cudf._lib.copying import get_element
from cudf.testing._utils import (
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
    assert_eq,
)


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[[]]],
        [[0]],
        [[0, 1]],
        [[0, 1], [2, 3]],
        [[[0, 1], [2]], [[3, 4]]],
        [[None]],
        [[[None]]],
        [[None], None],
        [[1, None], [1]],
        [[1, None], None],
        [[[1, None], None], None],
    ],
)
def test_create_list_series(data):
    expect = pd.Series(data)
    got = cudf.Series(data)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[]]},
        {"a": [[None]]},
        {"a": [[1, 2, 3]]},
        {"a": [[1, 2, 3]], "b": [[2, 3, 4]]},
        {"a": [[1, 2, 3, None], [None]], "b": [[2, 3, 4], [5]], "c": None},
        {"a": [[1]], "b": [[1, 2, 3]]},
        pd.DataFrame({"a": [[1, 2, 3]]}),
    ],
)
def test_df_list_dtypes(data):
    expect = pd.DataFrame(data)
    got = cudf.DataFrame(data)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[[]]],
        [[0]],
        [[0, 1]],
        [[0, 1], [2, 3]],
        [[[0, 1], [2]], [[3, 4]]],
        [[[0, 1, None], None], None, [[3, 2, None], None]],
        [[["a", "c", None], None], None, [["b", "d", None], None]],
    ],
)
def test_leaves(data):
    pa_array = pa.array(data)
    while hasattr(pa_array, "flatten"):
        pa_array = pa_array.flatten()
    dtype = "int8" if isinstance(pa_array, pa.NullArray) else None
    expect = cudf.Series(pa_array, dtype=dtype)
    got = cudf.Series(data).list.leaves
    assert_eq(expect, got)


def test_list_to_pandas_nullable_true():
    df = cudf.DataFrame({"a": cudf.Series([[1, 2, 3]])})
    actual = df.to_pandas(nullable=True)
    expected = pd.DataFrame({"a": pd.Series([[1, 2, 3]])})

    assert_eq(actual, expected)


def test_listdtype_hash():
    a = cudf.core.dtypes.ListDtype("int64")
    b = cudf.core.dtypes.ListDtype("int64")

    assert hash(a) == hash(b)

    c = cudf.core.dtypes.ListDtype("int32")

    assert hash(a) != hash(c)


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[1, 2, 3], [4, 5]],
        [[1, 2, 3], [], [4, 5]],
        [[1, 2, 3], None, [4, 5]],
        [[None, None], [None]],
        [[[[[[1, 2, 3]]]]]],
        cudf.Series([[1, 2]]).iloc[0:0],
        cudf.Series([None, [1, 2]]).iloc[0:1],
    ],
)
def test_len(data):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas()

    expect = psr.map(lambda x: len(x) if x is not None else None)
    got = gsr.list.len()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    ("data", "idx"),
    [
        ([[1, 2, 3], [3, 4, 5], [4, 5, 6]], [[0, 1], [2], [1, 2]]),
        ([[1, 2, 3], [3, 4, 5], [4, 5, 6]], [[1, 2, 0], [1, 0, 2], [0, 1, 2]]),
        ([[1, 2, 3], []], [[0, 1], []]),
        ([[1, 2, 3], [None]], [[0, 1], []]),
        ([[1, None, 3], None], [[0, 1], []]),
    ],
)
def test_take(data, idx):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.Series(zip(ps, idx)).map(
        lambda x: [x[0][i] for i in x[1]] if x[0] is not None else None
    )
    got = gs.list.take(idx)
    assert_eq(expected, got)


@pytest.mark.parametrize(
    ("invalid", "exception"),
    [
        ([[0]], pytest.raises(ValueError, match="different size")),
        ([1, 2, 3, 4], pytest.raises(ValueError, match="should be list type")),
        (
            [["a", "b"], ["c"]],
            pytest.raises(
                TypeError, match="should be column of values of index types"
            ),
        ),
        (
            [[[1], [0]], [[0]]],
            pytest.raises(
                TypeError, match="should be column of values of index types"
            ),
        ),
        ([[0, 1], None], pytest.raises(ValueError, match="contains null")),
    ],
)
def test_take_invalid(invalid, exception):
    gs = cudf.Series([[0, 1], [2, 3]])
    with exception:
        gs.list.take(invalid)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([[1, 1, 2, 2], [], None, [3, 4, 5]], [[1, 2], [], None, [3, 4, 5]]),
        (
            [[1.233, np.nan, 1.234, 3.141, np.nan, 1.234]],
            [[1.233, 1.234, np.nan, 3.141]],
        ),  # duplicate nans
        ([[1, 1, 2, 2, None, None]], [[1, 2, None]]),  # duplicate nulls
        (
            [[1.233, np.nan, None, 1.234, 3.141, np.nan, 1.234, None]],
            [[1.233, 1.234, np.nan, None, 3.141]],
        ),  # duplicate nans and nulls
        ([[2, None, 1, None, 2]], [[1, 2, None]]),
        ([[], []], [[], []]),
        ([[], None], [[], None]),
    ],
)
def test_unique(data, expected):
    """
    Pandas de-duplicates nans and nulls respectively in Series.unique.
    `expected` is setup to mimic such behavior
    """
    gs = cudf.Series(data, nan_as_null=False)

    got = gs.list.unique()
    expected = cudf.Series(expected, nan_as_null=False).list.sort_values()

    got = got.list.sort_values()

    assert_eq(expected, got)


def key_func_builder(x, na_position):
    if x is None:
        if na_position == "first":
            return -1e8
        else:
            return 1e8
    else:
        return x


@pytest.mark.parametrize(
    "data",
    [
        [[4, 2, None, 9], [8, 8, 2], [2, 1]],
        [[4, 2, None, 9], [8, 8, 2], None],
        [[4, 2, None, 9], [], None],
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        None,
        pd.Index(["a", "b", "c"]),
        pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (1, "a")], names=["l0", "l1"]
        ),
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_sort_values(data, index, ascending, na_position, ignore_index):
    key_func = functools.partial(key_func_builder, na_position=na_position)

    ps = pd.Series(data, index=index)
    gs = cudf.from_pandas(ps)

    expected = ps.apply(
        lambda x: sorted(x, key=key_func, reverse=not ascending)
        if x is not None
        else None
    )
    if ignore_index:
        expected.reset_index(drop=True, inplace=True)
    got = gs.list.sort_values(
        ascending=ascending, na_position=na_position, ignore_index=ignore_index
    )

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data, index, expect",
    [
        ([[None, None], [None, None]], 0, [None, None]),
        ([[1, 2], [3, 4]], 0, [1, 3]),
        ([["a", "b"], ["c", "d"]], 1, ["b", "d"]),
        ([[1, None], [None, 2]], 1, [None, 2]),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 1, [[3, 4], [7, 8]]),
    ],
)
def test_get(data, index, expect):
    sr = cudf.Series(data)
    expect = cudf.Series(expect)
    got = sr.list.get(index)
    assert_eq(expect, got)


def test_get_nested_lists():
    sr = cudf.Series(
        [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [], [[3, 4], [7, 8]]],
            [[], [[9, 10]], [[11, 12], [13, 14]]],
        ]
    )
    expect = cudf.Series([[[1, 2], [3, 4]], []])
    got = sr.list.get(0)
    assert_eq(expect, got)


def test_get_nulls():
    with pytest.raises(IndexError, match="list index out of range"):
        sr = cudf.Series([[], [], []])
        sr.list.get(100)


@pytest.mark.parametrize(
    "data, scalar, expect",
    [
        ([[1, 2, 3], []], 1, [True, False],),
        ([[1, 2, 3], [], [3, 4, 5]], 6, [False, False, False],),
        ([[1.0, 2.0, 3.0], None, []], 2.0, [True, None, False],),
        ([[None, "b", "c"], [], ["b", "e", "f"]], "b", [True, False, True],),
        ([[None, 2, 3], None, []], 1, [None, None, False]),
        ([[None, "b", "c"], [], ["b", "e", "f"]], "d", [None, False, False],),
    ],
)
def test_contains_scalar(data, scalar, expect):
    sr = cudf.Series(data)
    expect = cudf.Series(expect)
    got = sr.list.contains(cudf.Scalar(scalar, sr.dtype.element_type))
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data, expect",
    [
        ([[1, 2, 3], []], [None, None],),
        ([[1.0, 2.0, 3.0], None, []], [None, None, None],),
        ([[None, 2, 3], [], None], [None, None, None],),
        ([[1, 2, 3], [3, 4, 5]], [None, None],),
        ([[], [], []], [None, None, None],),
    ],
)
def test_contains_null_search_key(data, expect):
    sr = cudf.Series(data)
    expect = cudf.Series(expect, dtype="bool")
    got = sr.list.contains(cudf.Scalar(cudf.NA, sr.dtype.element_type))
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "row",
    [
        [[]],
        [[1]],
        [[1, 2]],
        [[1, 2], [3, 4, 5]],
        [[1, 2], [], [3, 4, 5]],
        [[1, 2, None], [3, 4, 5]],
        [[1, 2, None], None, [3, 4, 5]],
        [[1, 2, None], None, [], [3, 4, 5]],
        [[[1, 2], [3, 4]], [[5, 6, 7], [8, 9]]],
        [[["a", "c", "de", None], None, ["fg"]], [["abc", "de"], None]],
    ],
)
@pytest.mark.parametrize("dropna", [True, False])
def test_concat_elements(row, dropna):
    if any(x is None for x in row):
        if dropna:
            row = [x for x in row if x is not None]
            result = functools.reduce(operator.add, row)
        else:
            result = None
    else:
        result = functools.reduce(operator.add, row)

    expect = pd.Series([result])
    got = cudf.Series([row]).list.concat(dropna=dropna)
    assert_eq(expect, got)


def test_concat_elements_raise():
    s = cudf.Series([[1, 2, 3]])  # no nesting
    with pytest.raises(ValueError):
        s.list.concat()


def test_concatenate_rows_of_lists():
    pdf = pd.DataFrame({"val": [["a", "a"], ["b"], ["c"]]})
    gdf = cudf.from_pandas(pdf)

    expect = pdf["val"] + pdf["val"]
    got = gdf["val"] + gdf["val"]

    assert_eq(expect, got)


def test_concatenate_list_with_nonlist():
    with pytest.raises(TypeError, match="can only concatenate list to list"):
        gdf1 = cudf.DataFrame({"A": [["a", "c"], ["b", "d"], ["c", "d"]]})
        gdf2 = cudf.DataFrame({"A": ["a", "b", "c"]})
        gdf1["A"] + gdf2["A"]


@pytest.mark.parametrize(
    "data",
    [
        [1],
        [1, 2, 3],
        [[1, 2, 3], [4, 5, 6]],
        [NA],
        [1, NA, 3],
        [[1, NA, 3], [NA, 5, 6]],
    ],
)
def test_list_getitem(data):
    list_sr = cudf.Series([data])
    assert list_sr[0] == data


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [[1, 2, 3], [4, 5, 6]],
        ["a", "b", "c"],
        [["a", "b", "c"], ["d", "e", "f"]],
        [1.1, 2.2, 3.3],
        [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]],
        [1, NA, 3],
        [[1, NA, 3], [4, 5, NA]],
        ["a", NA, "c"],
        [["a", NA, "c"], ["d", "e", NA]],
        [1.1, NA, 3.3],
        [[1.1, NA, 3.3], [4.4, 5.5, NA]],
    ],
)
def test_list_scalar_host_construction(data):
    slr = cudf.Scalar(data)
    assert slr.value == data
    assert slr.device_value.value == data


@pytest.mark.parametrize(
    "elem_type", NUMERIC_TYPES + DATETIME_TYPES + TIMEDELTA_TYPES + ["str"]
)
@pytest.mark.parametrize("nesting_level", [1, 2, 3])
def test_list_scalar_host_construction_null(elem_type, nesting_level):
    dtype = cudf.ListDtype(elem_type)
    for level in range(nesting_level - 1):
        dtype = cudf.ListDtype(dtype)

    slr = cudf.Scalar(None, dtype=dtype)
    assert slr.value is cudf.NA


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [[1, 2, 3], [4, 5, 6]],
        ["a", "b", "c"],
        [["a", "b", "c"], ["d", "e", "f"]],
        [1.1, 2.2, 3.3],
        [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]],
        [1, NA, 3],
        [[1, NA, 3], [4, 5, NA]],
        ["a", NA, "c"],
        [["a", NA, "c"], ["d", "e", NA]],
        [1.1, NA, 3.3],
        [[1.1, NA, 3.3], [4.4, 5.5, NA]],
    ],
)
def test_list_scalar_device_construction(data):
    col = cudf.Series([data])._column
    slr = get_element(col, 0)
    assert slr.value == data


@pytest.mark.parametrize("nesting_level", [1, 2, 3])
def test_list_scalar_device_construction_null(nesting_level):
    data = [[]]
    for i in range(nesting_level - 1):
        data = [data]

    arrow_type = pa.infer_type(data)
    arrow_arr = pa.array([None], type=arrow_type)

    col = cudf.Series(arrow_arr)._column
    slr = get_element(col, 0)

    assert slr.value is cudf.NA


@pytest.mark.parametrize("input_obj", [[[1, NA, 3]], [[1, NA, 3], [4, 5, NA]]])
def test_construction_series_with_nulls(input_obj):
    expect = pa.array(input_obj, from_pandas=True)
    got = cudf.Series(input_obj).to_arrow()

    assert expect == got


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[]]},
        {"a": [[1, 2, None, 4]]},
        {"a": [["cat", None, "dog"]]},
        {
            "a": [[1, 2, 3, None], [4, None, 5]],
            "b": [None, ["fish", "bird"]],
            "c": [[], []],
        },
        {"a": [[1, 2, 3, None], [4, None, 5], None, [6, 7]]},
    ],
)
def test_serialize_list_columns(data):
    df = cudf.DataFrame(data)
    recreated = df.__class__.deserialize(*df.serialize())
    assert_eq(recreated, df)


@pytest.mark.parametrize(
    "data,item",
    [
        (
            # basic list into a list column
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [0, 0, 0],
        ),
        (
            # nested list into nested list column
            [
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
            ],
            [[0, 0, 0], [0, 0, 0]],
        ),
        (
            # NA into a list column
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            NA,
        ),
        (
            # NA into nested list column
            [
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
            ],
            NA,
        ),
    ],
)
def test_listcol_setitem(data, item):
    sr = cudf.Series(data)

    sr[1] = item
    data[1] = item
    expect = cudf.Series(data)

    assert_eq(expect, sr)


@pytest.mark.parametrize(
    "data,item,error",
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 3], [4, 5, 6]],
            "list nesting level mismatch",
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            0,
            "Can not set 0 into ListColumn",
        ),
    ],
)
def test_listcol_setitem_error_cases(data, item, error):
    sr = cudf.Series(data)
    with pytest.raises(BaseException, match=error):
        sr[1] = item
