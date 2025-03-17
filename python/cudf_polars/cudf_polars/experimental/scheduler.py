# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Synchronous task scheduler."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping
    from typing import TypeAlias


Key: TypeAlias = str | int | tuple["Key", ...]
Graph: TypeAlias = MutableMapping[Key, Any]


# NOTE: This is a slimmed-down version of the single-threaded
# (synchronous) scheduler in `dask.core`.


def ishashable(x):
    """Check if x is hashable."""
    try:
        hash(x)
    except TypeError:
        return False
    else:
        return True


def istask(x):
    """Check if x is a runnable task."""
    return type(x) is tuple and x and callable(x[0])


def _execute_task(arg, cache: MutableMapping):
    """Collecting data and execute a function."""
    if isinstance(arg, list):
        return [_execute_task(a, cache) for a in arg]
    elif istask(arg):
        func, args = arg[0], arg[1:]
        # Note: Don't assign the subtask results to a variable. numpy detects
        # temporaries by their reference count and can execute certain
        # operations in-place.
        return func(*(_execute_task(a, cache) for a in args))
    elif not ishashable(arg):
        return arg
    elif arg in cache:
        return cache[arg]
    else:
        return arg


def get(graph: Graph, key: Key, cache: MutableMapping | None = None):
    """Execute the task graph for a given key."""
    if key not in graph:
        raise KeyError(f"{key} is not a key in the graph")
    if cache is None:
        cache = {}
    for key in toposort(graph):
        task = graph[key]
        result = _execute_task(task, cache)
        cache[key] = result
    return _execute_task(key, cache)


def keys_in_tasks(keys: Collection[Key], tasks: Iterable[Any]):
    """Returns the keys in `keys` that are also in `tasks`."""
    ret = []
    while tasks:
        work = []
        for w in tasks:
            typ = type(w)
            if typ is tuple and w and callable(w[0]):  # istask(w)
                work.extend(w[1:])
            elif typ is list:
                work.extend(w)
            elif typ is dict:
                work.extend(w.values())
            else:
                try:
                    if w in keys:
                        ret.append(w)
                except TypeError:  # not hashable
                    pass
        tasks = work
    return ret


T_ = TypeVar("T_")


def _reverse_dict(d: Mapping[T_, Iterable[T_]]) -> dict[T_, set[T_]]:
    result: defaultdict[T_, set[T_]] = defaultdict(set)
    _add = set.add
    for k, vals in d.items():
        result[k]
        for val in vals:
            _add(result[val], k)
    return dict(result)


def toposort(graph):
    """Return a list of task keys sorted in topological order."""
    # Stack-based depth-first search traversal.  This is based on Tarjan's
    # method for topological sorting (see wikipedia for pseudocode)
    keys = graph
    ordered = []

    # Nodes whose descendents have been completely explored.
    # These nodes are guaranteed to not be part of a cycle.
    completed = set()

    # All nodes that have been visited in the current traversal.  Because
    # we are doing depth-first search, going "deeper" should never result
    # in visiting a node that has already been seen.  The `seen` and
    # `completed` sets are mutually exclusive; it is okay to visit a node
    # that has already been added to `completed`.
    seen = set()

    dependencies = {k: keys_in_tasks(graph, [graph[k]]) for k in graph}
    for key in keys:
        if key in completed:
            continue
        nodes = [key]
        while nodes:
            # Keep current node on the stack until all descendants are visited
            cur = nodes[-1]
            if cur in completed:
                # Already fully traversed descendants of cur
                nodes.pop()
                continue
            seen.add(cur)

            # Add direct descendants of cur to nodes stack
            next_nodes = []
            for nxt in dependencies[cur]:
                if nxt not in completed:
                    if nxt in seen:
                        # Cycle detected!
                        # Let's report only the nodes that directly participate in the cycle.
                        # We use `priorities` below to greedily construct a short cycle.
                        # Shorter cycles may exist.
                        priorities = {}
                        prev = nodes[-1]
                        # Give priority to nodes that were seen earlier.
                        while nodes[-1] != nxt:
                            priorities[nodes.pop()] = -len(priorities)
                        priorities[nxt] = -len(priorities)
                        # We're going to get the cycle by walking backwards along dependents,
                        # so calculate dependents only for the nodes in play.
                        inplay = set(priorities)
                        dependents = _reverse_dict(
                            {k: inplay.intersection(dependencies[k]) for k in inplay}
                        )
                        # Begin with the node that was seen twice and the node `prev` from
                        # which we detected the cycle.
                        cycle = [nodes.pop()]
                        cycle.append(prev)
                        while prev != cycle[0]:
                            # Greedily take a step that takes us closest to completing the cycle.
                            # This may not give us the shortest cycle, but we get *a* short cycle.
                            deps = dependents[cycle[-1]]
                            prev = min(deps, key=priorities.__getitem__)
                            cycle.append(prev)
                        cycle.reverse()
                        cycle = "->".join(str(x) for x in cycle)
                        raise RuntimeError(f"Cycle detected in the graph: {cycle}")
                    next_nodes.append(nxt)

            if next_nodes:
                nodes.extend(next_nodes)
            else:
                # cur has no more descendants to explore, so we're done with it
                ordered.append(cur)
                completed.add(cur)
                seen.remove(cur)
                nodes.pop()

    return ordered
