# Copyright (c) 2025, NVIDIA CORPORATION.


import pandas as pd

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
