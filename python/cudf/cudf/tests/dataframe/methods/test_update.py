# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize(
    "left_keys,right_keys",
    [
        [("a", "b"), ("a", "b")],
        [("a", "b"), ("a", "c")],
        [("a", "b"), ("d", "e")],
    ],
)
@pytest.mark.parametrize(
    "data_left,data_right",
    [
        [([1, 2, 3], [3, 4, 5]), ([1, 2, 3], [3, 4, 5])],
        [
            ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]),
            ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]),
        ],
        [
            ([True, False, True], [False, False, False]),
            ([True, False, True], [False, False, False]),
        ],
        [
            ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]),
            ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]),
        ],
        [([1, 2, 3], [3, 4, 5]), ([1, 2, 4], [30, 40, 50])],
        [
            ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]),
            ([1.0, 2.0, 4.0], [30.0, 40.0, 50.0]),
        ],
        [([1, 2, 3], [3, 4, 5]), ([10, 20, 40], [30, 40, 50])],
        [
            ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]),
            ([10.0, 20.0, 40.0], [30.0, 40.0, 50.0]),
        ],
    ],
)
def test_update_for_dataframes(
    left_keys, right_keys, data_left, data_right, overwrite
):
    errors = "ignore"
    join = "left"
    left = dict(zip(left_keys, data_left, strict=True))
    right = dict(zip(right_keys, data_right, strict=True))
    pdf = pd.DataFrame(left)
    gdf = cudf.DataFrame(left, nan_as_null=False)

    other_pd = pd.DataFrame(right)
    other_gd = cudf.DataFrame(right, nan_as_null=False)

    pdf.update(other=other_pd, join=join, overwrite=overwrite, errors=errors)
    gdf.update(other=other_gd, join=join, overwrite=overwrite, errors=errors)

    assert_eq(pdf, gdf, check_dtype=False)


def test_update_for_right_join():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})
    other_gd = cudf.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2.0, 5.0]})

    with pytest.raises(
        NotImplementedError, match="Only left join is supported"
    ):
        gdf.update(other_gd, join="right")


def test_update_for_data_overlap():
    errors = "raise"
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})

    other_pd = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2.0, 5.0]})
    other_gd = cudf.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2.0, 5.0]})

    assert_exceptions_equal(
        lfunc=pdf.update,
        rfunc=gdf.update,
        lfunc_args_and_kwargs=([other_pd, errors], {}),
        rfunc_args_and_kwargs=([other_gd, errors], {}),
    )
