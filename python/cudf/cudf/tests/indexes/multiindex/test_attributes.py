# Copyright (c) 2025, NVIDIA CORPORATION.


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.fixture(
    params=[
        "from_product",
        "from_tuples",
        "from_arrays",
        "init",
    ]
)
def midx(request):
    if request.param == "from_product":
        return cudf.MultiIndex.from_product([[0, 1], [1, 0]])
    elif request.param == "from_tuples":
        return cudf.MultiIndex.from_tuples([(0, 1), (0, 0), (1, 1), (1, 0)])
    elif request.param == "from_arrays":
        return cudf.MultiIndex.from_arrays([[0, 0, 1, 1], [1, 0, 1, 0]])
    elif request.param == "init":
        return cudf.MultiIndex(
            levels=[[0, 1], [0, 1]], codes=[[0, 0, 1, 1], [1, 0, 1, 0]]
        )
    else:
        raise NotImplementedError(f"{request.param} not implemented")


def test_multindex_constructor_levels_always_indexes(midx):
    assert_eq(midx.levels[0], cudf.Index([0, 1]))
    assert_eq(midx.levels[1], cudf.Index([0, 1]))


def test_bool_raises():
    assert_exceptions_equal(
        lfunc=bool,
        rfunc=bool,
        lfunc_args_and_kwargs=[[cudf.MultiIndex.from_arrays([range(1)])]],
        rfunc_args_and_kwargs=[[pd.MultiIndex.from_arrays([range(1)])]],
    )


def test_multi_index_contains_hashable():
    gidx = cudf.MultiIndex.from_tuples(zip(["foo", "bar", "baz"], [1, 2, 3]))
    pidx = gidx.to_pandas()

    assert_exceptions_equal(
        lambda: [] in gidx,
        lambda: [] in pidx,
        lfunc_args_and_kwargs=((),),
        rfunc_args_and_kwargs=((),),
    )


def test_multiindex_codes():
    midx = cudf.MultiIndex.from_tuples(
        [("a", "b"), ("a", "c"), ("b", "c")], names=["A", "Z"]
    )

    for p_array, g_array in zip(midx.to_pandas().codes, midx.codes):
        assert_eq(p_array, g_array)


def test_multiindex_values_pandas_compatible():
    midx = cudf.MultiIndex.from_tuples([(10, 12), (8, 9), (3, 4)])
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            midx.values


@pytest.mark.parametrize("bad", ["foo", ["foo"]])
def test_multiindex_set_names_validation(bad):
    mi = cudf.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    with pytest.raises(ValueError):
        mi.names = bad
