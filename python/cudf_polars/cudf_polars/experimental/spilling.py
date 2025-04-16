# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Spilling in multi-partition Dask execution using RAPIDSMPF."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, overload

from rapidsmpf.integrations.dask.spilling import SpillableWrapper

from cudf_polars.containers import DataFrame

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from typing import Any


T = TypeVar("T")


class _Callable[T](Protocol):
    """Generic callable."""

    def __call__(self, *args: Any) -> T: ...


@overload
def wrap_dataframe(obj: DataFrame) -> SpillableWrapper[DataFrame]: ...


@overload
def wrap_dataframe(obj: T) -> T: ...


def wrap_dataframe(obj: DataFrame | T) -> SpillableWrapper[DataFrame] | T:
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


def unwrap_arg(obj: SpillableWrapper[T] | T) -> T:
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
def wrap_func_spillable(
    func: _Callable[DataFrame], *, make_func_output_spillable: Literal[True]
) -> _Callable[SpillableWrapper[DataFrame]]: ...


@overload
def wrap_func_spillable(
    func: _Callable[T], *, make_func_output_spillable: bool
) -> _Callable[T]: ...


def wrap_func_spillable(
    func: _Callable[T] | _Callable[DataFrame],
    *,
    make_func_output_spillable: bool,
) -> _Callable[T] | _Callable[SpillableWrapper[DataFrame]]:
    """
    Wraps a function to handle spillable DataFrames.

    Parameters
    ----------
    func
        The function to be wrapped.
    make_func_output_spillable
        Whether to wrap the function's output in a SpillableWrapper.

    Returns
    -------
    A wrapped function that processes spillable DataFrames.
    """

    def wrapper(*args: Any) -> T:
        ret: Any = func(*(unwrap_arg(arg) for arg in args))
        if make_func_output_spillable:
            ret = wrap_dataframe(ret)
        return ret

    return wrapper


def wrap_dataframe_in_spillable(
    graph: MutableMapping[Any, Any], ignore_key: str | tuple[str, int]
) -> MutableMapping[Any, Any]:
    """
    Wraps functions within a task graph to handle spillable DataFrames.

    Only supports flat task graphs where each DataFrame can be found in the
    outermost level. Currently, this is true for all cudf-polars task graphs.

    Parameters
    ----------
    graph
        Dask graph.
    ignore_key
        The key to ignore when wrapping function, typically the key of the
        output node.

    Returns
    -------
    A new task graph with wrapped functions.
    """
    ret = {}
    for key, task in graph.items():
        assert isinstance(task, tuple)
        ret[key] = tuple(
            wrap_func_spillable(a, make_func_output_spillable=key != ignore_key)
            if callable(a)
            else a
            for a in task
        )
    return ret
