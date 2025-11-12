# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from time import sleep

import pandas as pd
import pytest

from cudf.pandas.fast_slow_proxy import _FastSlowProxyMeta
from cudf.pandas.module_accelerator import disable_module_accelerator


def is_enabled(df: pd.DataFrame):
    return type(type(df)) is _FastSlowProxyMeta


def per_thread_work(_):
    assert is_enabled(pd.DataFrame())

    with disable_module_accelerator():
        assert not is_enabled(pd.DataFrame())

        # Do some fake work to allow other threads to potentially modify this one
        for _ in range(1000):
            sleep(1e-6)

        assert not is_enabled(pd.DataFrame())

        # Ensure that nesting the context manager works too
        with disable_module_accelerator():
            assert not is_enabled(pd.DataFrame())
            for _ in range(1000):
                sleep(1e-6)

            assert not is_enabled(pd.DataFrame())
        assert not is_enabled(pd.DataFrame())

    assert is_enabled(pd.DataFrame())


@pytest.mark.flaky(reruns=3, reruns_delay=30)
def test_disable_pandas_accelerator_multi_threaded():
    num_threads = 20
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in executor.map(per_thread_work, range(num_threads * 10)):
            pass
