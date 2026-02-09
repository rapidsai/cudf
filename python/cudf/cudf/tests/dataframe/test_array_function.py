# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.mean(x, axis=0),
        lambda x: np.sum(x, axis=0),
        lambda x: np.var(x, ddof=1, axis=0),
        lambda x: np.dot(x, x.transpose()),
        np.all,
        np.any,
        lambda x: np.prod(x, axis=0),
        lambda x: np.prod(x, axis=1),
    ],
)
def test_array_func_cudf_dataframe(func):
    pd_df = pd.DataFrame(np.ones((3, 3)))
    cudf_df = cudf.from_pandas(pd_df)
    expect = func(pd_df)
    got = func(cudf_df)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        np.linalg.norm,
        np.linalg.det,
    ],
)
def test_array_func_missing_cudf_dataframe(func):
    pd_df = pd.DataFrame(np.ones((3, 3)))
    cudf_df = cudf.from_pandas(pd_df)
    with pytest.raises(TypeError):
        func(cudf_df)
