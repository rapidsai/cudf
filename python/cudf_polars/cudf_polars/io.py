# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for IO operations in cudf-polars."""

from __future__ import annotations

import concurrent.futures
import functools


@functools.cache
def io_threadpool() -> concurrent.futures.ThreadPoolExecutor:
    """
    A reusable thread pool, used for parallelizing IO operations in cudf-polars.

    Notes
    -----
    This is cached for the lifetime of the process. Do not shut this down. Other
    threads may still be involved in IO (both metadata and data operations) from
    libcudf or kvikio.
    """
    return concurrent.futures.ThreadPoolExecutor()
