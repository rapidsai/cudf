# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_multiindex_levels_codes_validation():
    levels = [["a", "b"], ["c", "d"]]

    # Codes not a sequence of sequences
    assert_exceptions_equal(
        lfunc=pd.MultiIndex,
        rfunc=cudf.MultiIndex,
        lfunc_args_and_kwargs=([levels, [0, 1]],),
        rfunc_args_and_kwargs=([levels, [0, 1]],),
    )

    # Codes don't match levels
    assert_exceptions_equal(
        lfunc=pd.MultiIndex,
        rfunc=cudf.MultiIndex,
        lfunc_args_and_kwargs=([levels, [[0], [1], [1]]],),
        rfunc_args_and_kwargs=([levels, [[0], [1], [1]]],),
    )

    # Largest code greater than number of levels
    assert_exceptions_equal(
        lfunc=pd.MultiIndex,
        rfunc=cudf.MultiIndex,
        lfunc_args_and_kwargs=([levels, [[0, 1], [0, 2]]],),
        rfunc_args_and_kwargs=([levels, [[0, 1], [0, 2]]],),
    )

    # Unequal code lengths
    assert_exceptions_equal(
        lfunc=pd.MultiIndex,
        rfunc=cudf.MultiIndex,
        lfunc_args_and_kwargs=([levels, [[0, 1], [0]]],),
        rfunc_args_and_kwargs=([levels, [[0, 1], [0]]],),
    )
    # Didn't pass levels and codes
    assert_exceptions_equal(lfunc=pd.MultiIndex, rfunc=cudf.MultiIndex)

    # Didn't pass non zero levels and codes
    assert_exceptions_equal(
        lfunc=pd.MultiIndex,
        rfunc=cudf.MultiIndex,
        lfunc_args_and_kwargs=([[], []],),
        rfunc_args_and_kwargs=([[], []],),
    )


def test_multiindex_construction():
    levels = [["a", "b"], ["c", "d"]]
    codes = [[0, 1], [1, 0]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels=levels, codes=codes)
    assert_eq(pmi, mi)


def test_multiindex_types():
    codes = [[0, 1], [1, 0]]
    levels = [[0, 1], [2, 3]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)
    levels = [[1.2, 2.1], [1.3, 3.1]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)
    levels = [["a", "b"], ["c", "d"]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)


def test_multiindex_from_tuples():
    arrays = [["a", "a", "b", "b"], ["house", "store", "house", "store"]]
    tuples = list(zip(*arrays, strict=True))
    pmi = pd.MultiIndex.from_tuples(tuples)
    gmi = cudf.MultiIndex.from_tuples(tuples)
    assert_eq(pmi, gmi)


def test_multiindex_from_dataframe():
    pdf = pd.DataFrame(
        [["a", "house"], ["a", "store"], ["b", "house"], ["b", "store"]]
    )
    gdf = cudf.from_pandas(pdf)
    pmi = pd.MultiIndex.from_frame(pdf, names=["alpha", "location"])
    gmi = cudf.MultiIndex.from_frame(gdf, names=["alpha", "location"])
    assert_eq(pmi, gmi)


def test_multindex_from_frame_invalid():
    with pytest.raises(TypeError):
        cudf.MultiIndex.from_frame("invalid_input")


@pytest.mark.parametrize(
    "arrays",
    [
        [["a", "a", "b", "b"], ["house", "store", "house", "store"]],
        [["a", "n", "n"] * 10, ["house", "store", "house", "store"]],
        [
            ["a", "n", "n"],
            ["house", "store", "house", "store", "store"] * 10,
        ],
        [
            ["a", "a", "n"] * 50,
            ["house", "store", "house", "store", "store"] * 10,
        ],
    ],
)
def test_multiindex_from_product(arrays):
    pmi = pd.MultiIndex.from_product(arrays, names=["alpha", "location"])
    gmi = cudf.MultiIndex.from_product(arrays, names=["alpha", "location"])
    assert_eq(pmi, gmi)


@pytest.mark.parametrize(
    "array",
    [
        list,
        tuple,
        np.array,
        cp.array,
        pd.Index,
        cudf.Index,
        pd.Series,
        cudf.Series,
    ],
)
def test_multiindex_from_arrays(array):
    pd_data = [[0, 0, 1, 1], [1, 0, 1, 0]]
    cudf_data = [array(lst) for lst in pd_data]
    result = pd.MultiIndex.from_arrays(pd_data)
    expected = cudf.MultiIndex.from_arrays(cudf_data)
    assert_eq(result, expected)


@pytest.mark.parametrize("arg", ["foo", ["foo"]])
def test_multiindex_from_arrays_wrong_arg(arg):
    with pytest.raises(TypeError):
        cudf.MultiIndex.from_arrays(arg)


@pytest.mark.parametrize(
    "idx", [pd.Index, pd.CategoricalIndex, pd.DatetimeIndex, pd.TimedeltaIndex]
)
def test_from_arrays_infer_names(idx):
    arrays = [idx([1], name="foo"), idx([2], name="bar")]
    expected = pd.MultiIndex.from_arrays(arrays)
    result = cudf.MultiIndex.from_arrays(arrays)
    assert_eq(result, expected)


def test_multiindex_dtype_error():
    midx = cudf.MultiIndex.from_tuples([(10, 12), (8, 9), (3, 4)])
    with pytest.raises(TypeError):
        cudf.Index(midx, dtype="int64")
    with pytest.raises(TypeError):
        cudf.Index(midx.to_pandas(), dtype="int64")


def test_multiindex_duplicate_names():
    gi = cudf.MultiIndex(
        levels=[["a", "b"], ["b", "a"]],
        codes=[[0, 0], [0, 1]],
        names=["a", "a"],
    )
    pi = pd.MultiIndex(
        levels=[["a", "b"], ["b", "a"]],
        codes=[[0, 0], [0, 1]],
        names=["a", "a"],
    )

    assert_eq(gi, pi)


def test_multiindex_from_arrow():
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 1, 2, 3],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "c": np.array([1, 2, 3, None, 5], dtype="datetime64[s]"),
            "d": ["a", "b", "c", "d", "e"],
        }
    )
    pdf["a"] = pdf["a"].astype("category")
    ptb = pa.Table.from_pandas(pdf)
    gdi = cudf.MultiIndex.from_arrow(ptb)
    pdi = pd.MultiIndex.from_frame(pdf)

    assert_eq(pdi, gdi)
