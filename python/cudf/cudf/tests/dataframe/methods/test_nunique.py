# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "columns",
    [
        pd.RangeIndex(2, name="foo"),
        pd.MultiIndex.from_arrays([[1, 2], [2, 3]], names=["foo", 1]),
        pd.Index([3, 5], dtype=np.int8, name="foo"),
    ],
)
def test_nunique_preserve_column_in_index(columns):
    df = cudf.DataFrame([[1, 2]], columns=columns)
    result = df.nunique().index.to_pandas()
    assert_eq(result, columns, exact=True)


def test_dataframe_nunique():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [1, 1, 0]})
    pdf = gdf.to_pandas()

    actual = gdf.nunique()
    expected = pdf.nunique()

    assert_eq(expected, actual)
