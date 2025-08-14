# Copyright (c) 2025, NVIDIA CORPORATION.


import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_multiindex_transpose():
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        index=pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)]),
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.transpose(), gdf.transpose())
