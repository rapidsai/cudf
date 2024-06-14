# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import contextlib
import functools
import hashlib
import sys
from typing import Dict

import nvtx

import rmm.statistics

from cudf.options import get_option

_NVTX_COLORS = ["green", "blue", "purple", "rapids"]


def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


def _performance_tracking(func, domain="cudf_python"):
    """Decorator for applying performance tracking (if enabled)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            if get_option("memory_profiling"):
                rmm.statistics.enable_statistics()
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


def get_memory_records() -> (
    Dict[str, rmm.statistics.ProfilerRecords.MemoryRecord]
):
    """Get the memory records from the memory profiling

    Returns
    -------
    Dict that maps function names to memory records. Empty if
    memory profiling is disabled
    """
    return rmm.statistics.default_profiler_records.records


def print_memory_report(file=sys.stdout) -> None:
    """Pretty print the result fo the memory profiling

    Parameters
    ----------
    file
        The output stream
    """
    print(rmm.statistics.default_profiler_records.report(), file=file)
