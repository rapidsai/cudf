# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_from_pandas_function():
    pdf = pd.DataFrame({"x": [1, 2, 3]})
    gdf = cudf.from_pandas(pdf)
    assert isinstance(gdf, cudf.DataFrame)
    assert_eq(pdf, gdf)

    gdf = cudf.from_pandas(pdf.x)
    assert isinstance(gdf, cudf.Series)
    assert_eq(pdf.x, gdf)

    with pytest.raises(TypeError):
        cudf.from_pandas(123)
