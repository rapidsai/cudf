# Copyright (c) 2020-2025, NVIDIA CORPORATION.


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_concatenate_rows_of_lists():
    pdf = pd.DataFrame({"val": [["a", "a"], ["b"], ["c"]]})
    gdf = cudf.from_pandas(pdf)

    expect = pdf["val"] + pdf["val"]
    got = gdf["val"] + gdf["val"]

    assert_eq(expect, got)


def test_concatenate_list_with_nonlist():
    with pytest.raises(TypeError):
        gdf1 = cudf.DataFrame({"A": [["a", "c"], ["b", "d"], ["c", "d"]]})
        gdf2 = cudf.DataFrame({"A": ["a", "b", "c"]})
        gdf1["A"] + gdf2["A"]
