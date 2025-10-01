# Copyright (c) 2025, NVIDIA CORPORATION.


import cudf
from cudf.testing._utils import (
    assert_exceptions_equal,
)


def test_series_duplicate_index_reindex():
    gs = cudf.Series([0, 1, 2, 3], index=[0, 0, 1, 1])
    ps = gs.to_pandas()

    assert_exceptions_equal(
        gs.reindex,
        ps.reindex,
        lfunc_args_and_kwargs=([10, 11, 12, 13], {}),
        rfunc_args_and_kwargs=([10, 11, 12, 13], {}),
    )
