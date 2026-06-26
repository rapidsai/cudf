# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import functools
import hashlib
import sys

import nvtx

import rmm.statistics

from cudf.options import get_option

_NVTX_COLORS = ["green", "blue", "purple", "rapids"]
dummy_annotate = nvtx.annotate(
    "cudf_utils_performance_tracking", category="cudf_dummy"
)


def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


def _performance_tracking(func, domain="cudf_python"):
    """Decorator for applying performance tracking (if enabled)."""
    global dummy_annotate
    if dummy_annotate.domain is nvtx.nvtx.dummy_domain and not get_option(
        "memory_profiling"
    ):
        # If `dummy_domain` is enabled it means NVTX is present but nsys profiling is not enabled.
        # nsys profiler enabled (in which case dummy_annotate.domain is not a dummy_domain):
        # `nsys profile --trace=nvtx python script.py`
        # nsys profiler disabled (in which case dummy_annotate.domain is a dummy_domain):
        # `python script.py`
        # We also need to set the env variable `CUDF_MEMORY_PROFILING=1` prior to the launch of the Python
        # interpreter if `memory_profiling` is needed.
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            if get_option("memory_profiling"):
                # NB: the user still needs to call `rmm.statistics.enable_statistics()`
                #     to enable memory profiling.
                stack.enter_context(
                    rmm.statistics.profiler(
                        name=rmm.statistics._get_descriptive_name_of_object(
                            func
                        )
                    )
                )
            if nvtx.enabled():
                stack.enter_context(
                    nvtx.annotate(
                        message=func.__qualname__,
                        color=_get_color_for_nvtx(func.__qualname__),
                        domain=domain,
                    )
                )
            return func(*args, **kwargs)

    return wrapper


_dask_cudf_performance_tracking = functools.partial(
    _performance_tracking, domain="dask_cudf_python"
)


def get_memory_records() -> dict[
    str, rmm.statistics.ProfilerRecords.MemoryRecord
]:
    """Get the memory records from the memory profiling

    Returns
    -------
    Dict that maps function names to memory records. Empty if
    memory profiling is disabled
    """
    return rmm.statistics.default_profiler_records.records


def print_memory_report(file=sys.stdout) -> None:
    """Pretty print the result of the memory profiling

    Parameters
    ----------
    file
        The output stream
    """
    print(rmm.statistics.default_profiler_records.report(), file=file)
