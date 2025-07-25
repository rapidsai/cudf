# Copyright (c) 2025, NVIDIA CORPORATION.


import pytest

import cudf
from cudf.testing import assert_eq


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
