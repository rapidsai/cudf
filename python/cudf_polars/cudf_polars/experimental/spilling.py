# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Spilling in multi-partition Dask execution using RAPIDSMPF."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dask.sizeof import sizeof
from distributed import get_worker
from rapidsmpf.integrations.dask.core import get_worker_context
from rapidsmpf.integrations.dask.spilling import SpillableWrapper
from rapidsmpf.memory.buffer import MemoryType

from cudf_polars.containers import DataFrame

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping
    from typing import Any

    from cudf_polars.utils.config import ConfigOptions


def wrap_arg(obj: Any) -> Any:
    """
    Make `obj` spillable if it is a DataFrame.

    Parameters
    ----------
    obj
        The object to be wrapped (if it is a DataFrame).

    Returns
    -------
    A SpillableWrapper if obj is a DataFrame, otherwise the original object.
    """
    if isinstance(obj, DataFrame):
        return SpillableWrapper(on_device=obj)
    return obj


def unwrap_arg(obj: Any) -> Any:
    """
    Unwraps a SpillableWrapper to retrieve the original object.

    Parameters
    ----------
    obj
        The object to be unwrapped.

    Returns
    -------
    The unwrapped obj is a SpillableWrapper, otherwise the original object.
    """
    if isinstance(obj, SpillableWrapper):
        return obj.unspill()
    return obj


def wrap_func_spillable(
    func: Callable,
    *,
    make_func_output_spillable: bool,
    target_partition_size: int,
) -> Callable:
    """
    Wraps a function to handle spillable DataFrames.

    Parameters
    ----------
    func
        The function to be wrapped.
    make_func_output_spillable
        Whether to wrap the function's output in a SpillableWrapper.
    target_partition_size
        Target byte size for IO tasks.

    Returns
    -------
    A wrapped function that processes spillable DataFrames.
    """

    def wrapper(*args: Any) -> Any:
        # Make headroom before executing the task
        headroom = 0
        probable_io_task = True
        for arg in args:
            if isinstance(arg, SpillableWrapper):
                if arg.mem_type() == MemoryType.HOST:
                    headroom += sizeof(arg._on_host)
                probable_io_task = False
        if probable_io_task:
            # Likely an IO task - Assume we need target_partition_size
            headroom = target_partition_size
        if headroom > 128_000_000:  # Don't waste time on smaller data
            ctx = get_worker_context(get_worker())
            with ctx.lock:
                ctx.br.spill_manager.spill_to_make_headroom(headroom=headroom)

        ret: Any = func(*(unwrap_arg(arg) for arg in args))
        if make_func_output_spillable:
            ret = wrap_arg(ret)
        return ret

    return wrapper


def wrap_dataframe_in_spillable(
    graph: MutableMapping[Any, Any],
    ignore_key: str | tuple[str, int],
    config_options: ConfigOptions,
) -> MutableMapping[Any, Any]:
    """
    Wraps functions within a task graph to handle spillable DataFrames.

    Only supports flat task graphs where each DataFrame can be found in the
    outermost level. Currently, this is true for all cudf-polars task graphs.

    Parameters
    ----------
    graph
        Task graph.
    ignore_key
        The key to ignore when wrapping function, typically the key of the
        output node.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A new task graph with wrapped functions.
    """
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'wrap_dataframe_in_spillable'"
    )
    target_partition_size = config_options.executor.target_partition_size

    ret = {}
    for key, task in graph.items():
        assert isinstance(task, tuple)
        ret[key] = tuple(
            wrap_func_spillable(
                a,
                make_func_output_spillable=key != ignore_key,
                target_partition_size=target_partition_size,
            )
            if callable(a)
            else a
            for a in task
        )
    return ret
