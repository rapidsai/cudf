# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Spilling in multi-partition Dask execution using RAPIDSMPF."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import nvtx
from dask.sizeof import sizeof
from distributed import get_worker
from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.integrations.dask.core import get_worker_context
from rapidsmpf.integrations.dask.spilling import SpillableWrapper

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.tracing import CUDF_POLARS_NVTX_DOMAIN, do_evaluate_with_tracing

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping
    from typing import Any

    from cudf_polars.dsl.ir import IR
    from cudf_polars.utils.config import ConfigOptions


T = TypeVar("T")


@overload
def wrap_arg(obj: DataFrame) -> SpillableWrapper: ...


@overload
def wrap_arg(obj: T) -> T: ...


def wrap_arg(obj: T) -> T | SpillableWrapper:
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


# We can't really express this type correctly. We know that
# SpillableWrapper will always return a DataFrame, and all
# other types will be unchanged.
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


@overload
def do_evaluate_with_tracing_and_spilling(
    cls: type[IR],
    *args: Any,
    make_func_output_spillable: Literal[False],
    target_partition_size: int,
) -> DataFrame: ...


@overload
def do_evaluate_with_tracing_and_spilling(
    cls: type[IR],
    *args: Any,
    make_func_output_spillable: Literal[True],
    target_partition_size: int,
) -> SpillableWrapper: ...


def do_evaluate_with_tracing_and_spilling(
    cls: type[IR],
    *args: Any,
    make_func_output_spillable: bool,
    target_partition_size: int,
) -> DataFrame | SpillableWrapper:
    """
    Evaluate an IR node with tracing and spilling.

    Parameters
    ----------
    cls
        The type of the IR node to evaluate.
    args
        The arguments to pass to ``cls.do_evaluate``.
    make_func_output_spillable
        Whether to make the output of the function spillable.
    target_partition_size
        The target partition size.

    Returns
    -------
    output
        The result of calling ``cls.do_evaluate`` with the unwrapped arguments.
        If ``make_func_output_spillable`` is True, the output will be wrapped in
        a SpillableWrapper.
    """
    func = cls.do_evaluate

    with nvtx.annotate(
        message=cls.__name__,
        domain=CUDF_POLARS_NVTX_DOMAIN,
    ):
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


def unwrap_and_apply(func: Callable, *args: Any) -> Any:
    """
    Unwrap and SpillableWrapper arguments before calling ``func``.

    Parameters
    ----------
    func
        The function to be called.
    args
        The arguments to be passed to the function.

    Returns
    -------
    The result of calling ``func`` with the unwrapped arguments.
    """
    return func(*(unwrap_arg(arg) for arg in args))


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

    apply_and_spill = functools.partial(
        do_evaluate_with_tracing_and_spilling,
        make_func_output_spillable=True,
        target_partition_size=target_partition_size,
    )
    apply_no_spill = functools.partial(
        do_evaluate_with_tracing_and_spilling,
        make_func_output_spillable=False,
        target_partition_size=target_partition_size,
    )

    for key, task in graph.items():
        assert isinstance(task, tuple)

        if task and task[0] is do_evaluate_with_tracing:
            if key == ignore_key:
                new = apply_no_spill
            else:
                new = apply_and_spill

            ret[key] = (new, *task[1:])

        elif task and callable(task[0]):
            f, *args = task
            ret[key] = (unwrap_and_apply, f, *args)
        else:
            ret[key] = task

    return ret
