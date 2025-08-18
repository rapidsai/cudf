# Copyright (c) 2023-2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pytest

import cudf


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
    ],
)
def test_timedelta_series_to_numpy(data, timedelta_types_as_str):
    gsr = cudf.Series(data, dtype=timedelta_types_as_str)

    expected = np.array(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data,
        dtype=timedelta_types_as_str,
    )
    expected = expected[~np.isnan(expected)]

    actual = gsr.dropna().to_numpy()

    np.testing.assert_array_equal(expected, actual)
