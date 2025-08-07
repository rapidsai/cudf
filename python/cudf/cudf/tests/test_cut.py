# Copyright (c) 2021-2025, NVIDIA CORPORATION.

"""
Test related to Cut
"""

import numpy as np
import pandas as pd
import pytest

from cudf.core.cut import cut
from cudf.testing import assert_eq


@pytest.mark.parametrize("box", [list, np.array])
@pytest.mark.parametrize("bins", [1, 2, 3])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize("precision", [1, 3])
def test_cut_basic(box, bins, right, include_lowest, precision):
    # will test optional labels, retbins and duplicates separately
    # they need more specific parameters to work
    x = box([1, 7, 5, 4, 6, 3])
    ordered = True
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


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("precision", [1, 3])
@pytest.mark.parametrize(
    "labels", [["bad", "medium", "good"], [1, 2, 3], False]
)
def test_cut_labels(right, include_lowest, ordered, precision, labels):
    x = [1, 7, 5, 4, 6, 3]
    bins = 3
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


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize("precision", [1, 3])
@pytest.mark.parametrize("labels", [["bad", "good", "good"], [1, 2, 2], False])
def test_cut_labels_non_unique(right, include_lowest, precision, labels):
    x = [1, 7, 5, 4, 6, 3]
    bins = 3
    ordered = False
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
    "bins",
    [1, 2, 3, [1, 2, 3], [0, 2, 4, 6, 10]],
)
@pytest.mark.parametrize("right", [True, False])
def test_cut_right(x, bins, right):
    precision = 3
    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
    )
    pindex = pd.CategoricalIndex(pcat)
    gindex = cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
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
    "bins",
    [[0, 2, 4, 6, 10, 10], [1, 2, 2, 3, 3]],
)
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize("precision", [1, 3])
def test_cut_drop_duplicates(x, bins, right, precision, include_lowest):
    ordered = True
    duplicates = "drop"
    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        duplicates=duplicates,
        include_lowest=include_lowest,
        ordered=ordered,
    )
    pindex = pd.CategoricalIndex(pcat)
    gindex = cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        duplicates=duplicates,
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
    "bins",
    [[0, 2, 4, 6, 10, 10], [1, 2, 2, 3, 3]],
)
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize("precision", [1, 3])
def test_cut_drop_duplicates_raises(x, bins, right, precision, include_lowest):
    ordered = True
    duplicates = "raise"
    msg = "Bin edges must be unique"
    with pytest.raises(ValueError, match=msg):
        cut(
            x=x,
            bins=bins,
            right=right,
            precision=precision,
            duplicates=duplicates,
            include_lowest=include_lowest,
            ordered=ordered,
        )
    with pytest.raises(ValueError, match=msg):
        pd.cut(
            x=x,
            bins=bins,
            right=right,
            precision=precision,
            duplicates=duplicates,
            include_lowest=include_lowest,
            ordered=ordered,
        )


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
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("precision", [1, 3])
@pytest.mark.parametrize("duplicates", ["drop", "raise"])
def test_cut_intervalindex_bin(x, right, precision, duplicates):
    bins = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
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


@pytest.mark.parametrize("bins", [1, 3])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
def test_cut_series(bins, right, include_lowest):
    x = pd.Series(np.array([2, 4, 6, 8, 10]), index=["a", "b", "c", "d", "e"])
    precision = 3
    ordered = True
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
