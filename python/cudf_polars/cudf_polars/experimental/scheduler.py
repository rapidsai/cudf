# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Synchronous task scheduler."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypeAlias


Key: TypeAlias = str | tuple[str, int]
Graph: TypeAlias = MutableMapping[Key, Any]
T_ = TypeVar("T_")


# NOTE: This is a slimmed-down version of the single-threaded
# (synchronous) scheduler in `dask.core`.
#
# Key Differences:
# * We do not allow a task to contain a list of key names.
#   Keys must be distinct elements of the task.


def istask(x: Any) -> bool:
    """Check if x is a callable task."""
    return bool(type(x) is tuple and x and callable(x[0]))


def _execute_task(arg: Any, cache: MutableMapping) -> Any:
    """Execute a compute task."""
    if istask(arg):
        return arg[0](*(_execute_task(a, cache) for a in arg[1:]))
    else:
        try:
            return cache.get(arg, arg)
        except TypeError:  # Unhashable task argument
            return arg


def required_keys(key: Key, graph: Graph) -> list[Key]:
    """
    Return the dependencies to extract a key from the graph.

    Parameters
    ----------
    key
        Root key we want to extract.
    graph
        The full task graph.

    Returns
    -------
    List of other keys needed to extract ``key``.
    """
    keys = []
    tasks = [graph[key]]
    while tasks:
        work = []
        for w in tasks:
            if istask(w):
                work.extend(w[1:])
            else:
                try:
                    if w in graph:
                        keys.append(w)
                except TypeError:  # not hashable
                    pass
        tasks = work
    return keys


def toposort(graph: Graph) -> list[Key]:
    """Return a list of task keys sorted in topological order."""
    # Stack-based depth-first search traversal. This is based on Tarjan's
    # method for topological sorting (see wikipedia for pseudocode)
    ordered: list[Key] = []
    completed: set[Key] = set()
    seen: set[Key] = set()

    dependencies: Mapping[Key, list[Key]] = {k: required_keys(k, graph) for k in graph}
    for key in graph:
        if key in completed:
            continue
        nodes: list[Key] = [key]
        while nodes:
            # Keep current node on the stack until all descendants are visited
            current = nodes[-1]
            if current in completed:
                # Already fully traversed descendants of current
                nodes.pop()
                continue
            seen.add(current)

            # Add direct descendants of current to nodes stack
            next_nodes = []
            for nxt in dependencies[current]:
                if nxt not in completed:
                    if nxt in seen:  # pragma: no cover
                        # Cycle detected!
                        raise RuntimeError(
                            f"Cycle detected in the task graph! Key: {nxt}"
                        )
                    next_nodes.append(nxt)

            if next_nodes:
                nodes.extend(next_nodes)
            else:
                # Current has no more descendants to explore
                ordered.append(current)
                completed.add(current)
                seen.remove(current)
                nodes.pop()

    return ordered


def synchronous_scheduler(
    graph: Graph, key: Key, cache: MutableMapping | None = None
) -> Any:
    """
    Execute the task graph for a given key.

    Parameters
    ----------
    graph
        The task graph to execute.
    key
        The final output key to extract from the graph.
    cache
        Intermediate-data cache.

    Returns
    -------
    Executed task-graph result for ``key``.
    """
    if key not in graph:
        raise KeyError(f"{key} is not a key in the graph")
    if cache is None:
        cache = {}
    for k in toposort(graph):
        task = graph[k]
        result = _execute_task(task, cache)
        cache[k] = result
    return _execute_task(key, cache)
