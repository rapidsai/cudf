# Copyright (c) 2021, NVIDIA CORPORATION.

"""
Test related to Cut
"""

import pandas as pd
import numpy as np
from cudf.core.cut import cut
import pytest
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "x", [[1, 7, 5, 4, 6, 3], [1, 7], np.array([1, 7, 5, 4, 6, 3])]
)
@pytest.mark.parametrize("bins", [1, 2, 3])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize("ordered", [True])
@pytest.mark.parametrize("precision", [3])
def test_cut_basic(x, bins, right, include_lowest, ordered, precision):

    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )
    pindex = pd.CategoricalIndex(pcat)
    gindex = cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("x", [[1, 7, 5, 4, 6, 3]])
@pytest.mark.parametrize("bins", [3])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize("ordered", [False])
@pytest.mark.parametrize("precision", [3])
@pytest.mark.parametrize(
    "labels", [["bad", "medium", "good"], ["B", "A", "B"], False]
)
def test_cut_labels(
    x, bins, right, include_lowest, ordered, precision, labels
):

    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        labels=labels,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )
    pindex = pd.CategoricalIndex(pcat) if labels else pcat
    gindex = cut(
        x=x,
        bins=bins,
        right=right,
        labels=labels,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize(
    "x",
    [
        [1, 7, 5, 4, 6, 3],
        [1, 7],
        np.array([1, 7, 5, 4, 6, 3]),
        np.array([2, 4, 6, 8, 10]),
    ],
)
@pytest.mark.parametrize(
    "bins", [1, 2, 3, [1, 2, 3], [0, 2, 4, 6, 10]],
)
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("precision", [3])
def test_cut_right(x, bins, right, precision):

    pcat = pd.cut(x=x, bins=bins, right=right, precision=precision,)
    pindex = pd.CategoricalIndex(pcat)
    gindex = cut(x=x, bins=bins, right=right, precision=precision,)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize(
    "x",
    [
        [1, 7, 5, 4, 6, 3],
        [1, 7],
        np.array([1, 7, 5, 4, 6, 3]),
        np.array([2, 4, 6, 8, 10]),
    ],
)
@pytest.mark.parametrize(
    "bins", [1, 2, 3, [1, 2, 3], [0, 2, 4, 6, 10]],
)
@pytest.mark.parametrize("right", [False])
@pytest.mark.parametrize("precision", [3])
@pytest.mark.parametrize("duplicates", ["drop"])
def test_cut_drop_duplicates(x, bins, right, precision, duplicates):

    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        duplicates=duplicates,
    )
    pindex = pd.CategoricalIndex(pcat)
    gindex = cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        duplicates=duplicates,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize(
    "x",
    [
        [0, 0.5, 1.5, 2.5, 4.5],
        [1, 7, 5, 4, 6, 3],
        [1, 7],
        np.array([1, 7, 5, 4, 6, 3]),
        np.array([2, 4, 6, 8, 10]),
    ],
)
@pytest.mark.parametrize(
    "bins", [pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])],
)
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("precision", [3])
@pytest.mark.parametrize("duplicates", ["drop", "raise"])
def test_cut_intervalindex_bin(x, bins, right, precision, duplicates):

    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        duplicates=duplicates,
    )
    pindex = pd.CategoricalIndex(pcat)
    gindex = cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        duplicates=duplicates,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize(
    "x",
    [pd.Series(np.array([2, 4, 6, 8, 10]), index=["a", "b", "c", "d", "e"])],
)
@pytest.mark.parametrize("bins", [1, 2, 3])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize("ordered", [True])
@pytest.mark.parametrize("precision", [3])
def test_cut_series(x, bins, right, include_lowest, ordered, precision):

    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )

    gcat = cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )

    assert_eq(pcat, gcat)
