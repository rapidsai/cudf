# Copyright (c) 2024, NVIDIA CORPORATION.
from numpy.testing import assert_array_equal

import cudf
from cudf.testing import assert_eq


def test_groupby_14955():
    # https://github.com/rapidsai/cudf/issues/14955
    df = cudf.DataFrame({"a": [1, 2] * 2}, index=[0] * 4)
    agg = df.groupby("a")
    pagg = df.to_pandas().groupby("a")
    for key in agg.groups:
        assert_array_equal(pagg.indices[key], agg.indices[key].get())
        assert_eq(pagg.get_group(key), agg.get_group(key))
