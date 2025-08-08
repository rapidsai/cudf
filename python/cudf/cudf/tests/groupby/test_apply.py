# Copyright (c) 2018-2025, NVIDIA CORPORATION.


import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.tests.groupby.testing import assert_groupby_results_equal


@pytest.fixture(params=["cudf", "jit"])
def engine(request):
    return request.param


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Include groups missing on old versions of pandas",
)
def test_groupby_as_index_apply(as_index, engine):
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = gdf.groupby("y", as_index=as_index).apply(
        lambda df: df["x"].mean(), engine=engine
    )
    kwargs = {"func": lambda df: df["x"].mean(), "include_groups": False}
    pdf = pdf.groupby("y", as_index=as_index).apply(**kwargs)
    assert_groupby_results_equal(pdf, gdf, as_index=as_index, by="y")
