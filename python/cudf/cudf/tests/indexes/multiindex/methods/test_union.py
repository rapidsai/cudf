# Copyright (c) 2025, NVIDIA CORPORATION.


import cudf
from cudf.testing._utils import assert_exceptions_equal


def test_multiindex_union_error():
    midx = cudf.MultiIndex.from_tuples([(10, 12), (8, 9), (3, 4)])
    pidx = midx.to_pandas()

    assert_exceptions_equal(
        midx.union,
        pidx.union,
        lfunc_args_and_kwargs=(["a"],),
        rfunc_args_and_kwargs=(["b"],),
    )
