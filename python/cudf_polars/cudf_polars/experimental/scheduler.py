# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Synchronous task scheduler."""

from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import TYPE_CHECKING, Any

from cudf_polars.experimental.task import Key, Task, _task_deps

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.experimental.task import TaskGraph


# NOTE: This is a slimmed-down version of the single-threaded
# (synchronous) scheduler in `dask.core`.
#
# Key Differences:
# * We do not allow a task to contain a list of key names.
#   Keys must be distinct elements of the task.
# * We do not support nested tasks.


def toposort(graph: TaskGraph) -> list[Key]:
    """Return a list of task keys sorted in topological order."""
    # Stack-based depth-first search traversal. This is based on Tarjan's
    # algorithm for strongly-connected components
    # (https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
    ordered: list[Key] = []
    completed: set[Key] = set()

    for key in graph:
        if key in completed:
            continue
        nodes = [key]
        while nodes:
            # Keep current node on the stack until all descendants are visited
            current = nodes[-1]
            if current in completed:  # pragma: no cover
                # Already fully traversed descendants of current
                nodes.pop()
                continue

            # Add direct descendants of current to nodes stack
            if next_nodes := set(_task_deps(graph[current])) - completed:
                nodes.extend(next_nodes)
            else:
                # Current has no more descendants to explore
                ordered.append(current)
                completed.add(current)
                nodes.pop()

    return ordered


def synchronous_scheduler(
    graph: TaskGraph,
    key: Key,
    *,
    cache: MutableMapping | None = None,
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
    if key not in graph:  # pragma: no cover
        raise KeyError(f"{key} is not a key in the graph")
    if cache is None:
        cache = {}

    refcount = Counter(chain.from_iterable(_task_deps(val) for val in graph.values()))
    for k in toposort(graph):
        # Execute the task or retrieve the result from cache
        task_or_key = graph[k]
        if isinstance(task_or_key, Key):
            cache[k] = cache[task_or_key]
        elif isinstance(task_or_key, Task):
            cache[k] = task_or_key.execute(cache)
        else:  # pragma: no cover
            raise TypeError(f"Expected Key or Task, got {task_or_key}")

        # Clean up the cache
        for dep in _task_deps(task_or_key):
            refcount[dep] -= 1
            if refcount[dep] == 0 and dep != key:
                del cache[dep]

    return cache[key]
