# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings

import numpy as np
import pandas as pd
import pytest
import stumpy
from pandas._testing import assert_equal

from dask.distributed import Client, LocalCluster


def stumpy_assert_equal(expected, got):
    def as_float64(x):
        if isinstance(x, (tuple, list)):
            return [as_float64(y) for y in x]
        else:
            return x.astype(np.float64)

    assert_equal(as_float64(expected), as_float64(got))


pytestmark = pytest.mark.assert_eq(fn=stumpy_assert_equal)


# Shared dask client for all tests in this module
@pytest.fixture(scope="module")
def dask_client(worker_id: str):
    worker_count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0"))
    if worker_count > 0:
        # Avoid port conflicts with multiple test runners
        worker_index = int(worker_id.removeprefix("gw"))
        scheduler_port = 8800 + worker_index
        dashboard_address = 8900 + worker_index
    else:
        scheduler_port = None
        dashboard_address = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning, message="Port")

        with LocalCluster(
            n_workers=1,
            scheduler_port=scheduler_port,
            dashboard_address=dashboard_address,
        ) as cluster:
            with Client(cluster) as client:
                yield client


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_1d_distributed(dask_client):
    rng = np.random.default_rng(seed=42)
    ts = pd.Series(rng.random(100))
    m = 10
    return stumpy.stumped(dask_client, ts, m)


def test_multidimensional_distributed_timeseries(dask_client):
    rng = np.random.default_rng(seed=42)
    # Each row represents data from a different dimension while each column represents
    # data from the same dimension
    your_time_series = rng.random((3, 1000))
    # Approximately, how many data points might be found in a pattern
    window_size = 50

    return stumpy.mstumped(dask_client, your_time_series, m=window_size)
