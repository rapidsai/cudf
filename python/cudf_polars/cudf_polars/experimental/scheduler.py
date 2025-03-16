# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Synchronous task scheduler."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping
    from typing import TypeAlias


Key: TypeAlias = str | bytes | int | float | tuple["Key", ...]
Graph: TypeAlias = MutableMapping[Key, Any]


# NOTE: This is the single-threaded scheduler from `dask.core`.


class _NoDefault(Enum):
    """
    Unspecified key-word argument.

    Notes
    -----
    Typing-aware constant to detect when the user omits a parameter and you can't use
    None. Copied from pandas._libs.lib._NoDefault.
    """

    no_default = "NO_DEFAULT"

    def __repr__(self) -> str:
        return "<no_default>"


no_default = _NoDefault.no_default
NoDefault = Literal[_NoDefault.no_default]


def ishashable(x):
    """
    Check if x is hashable.

    Examples
    --------
    >>> ishashable(1)
    True
    >>> ishashable([1])
    False

    See Also
    --------
    iskey
    """
    try:
        hash(x)
    except TypeError:
        return False
    else:
        return True


def istask(x):
    """
    Check if x is a runnable task.

    A task is a tuple with a callable first argument.

    Examples
    --------
    >>> inc = lambda x: x + 1
    >>> istask((inc, 1))
    True
    >>> istask(1)
    False
    """
    return type(x) is tuple and x and callable(x[0])


def has_tasks(dsk, x):
    """
    Whether ``x`` has anything to compute.

    Returns True if:
    - ``x`` is a task
    - ``x`` is a key in ``dsk``
    - ``x`` is a list that contains any tasks or keys
    """
    if istask(x):
        return True
    try:
        if x in dsk:
            return True
    except Exception:
        pass
    if isinstance(x, list):
        for i in x:
            if has_tasks(dsk, i):
                return True
    return False


def preorder_traversal(task):
    """A generator to preorder-traverse a task."""
    for item in task:
        if istask(item):
            yield from preorder_traversal(item)
        elif isinstance(item, list):
            yield list
            yield from preorder_traversal(item)
        else:
            yield item


def _lists_to_tuples(res, keys):
    if isinstance(keys, list):
        return tuple(_lists_to_tuples(r, k) for r, k in zip(res, keys, strict=False))
    return res


def _execute_task(arg, cache, dsk=None):
    """
    Do the actual work of collecting data and executing a function.

    Examples
    --------
    >>> inc = lambda x: x + 1
    >>> add = lambda x, y: x + y
    >>> cache = {"x": 1, "y": 2}

    Compute tasks against a cache
    >>> _execute_task((add, "x", 1), cache)  # Compute task in naive manner
    2
    >>> _execute_task((add, (inc, "x"), 1), cache)  # Support nested computation
    3

    Also grab data from cache
    >>> _execute_task("x", cache)
    1

    Support nested lists
    >>> list(_execute_task(["x", "y"], cache))
    [1, 2]

    >>> list(map(list, _execute_task([["x", "y"], ["y", "x"]], cache)))
    [[1, 2], [2, 1]]

    >>> _execute_task("foo", cache)  # Passes through on non-keys
    'foo'
    """
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


def get(dsk, out, cache=None):
    """
    Get value from Dask.

    Examples
    --------
    >>> inc = lambda x: x + 1
    >>> d = {"x": 1, "y": (inc, "x")}

    >>> get(d, "x")
    1
    >>> get(d, "y")
    2
    """
    for k in flatten(out) if isinstance(out, list) else [out]:
        if k not in dsk:
            raise KeyError(f"{k} is not a key in the graph")
    if cache is None:
        cache = {}
    for key in toposort(dsk):
        task = dsk[key]
        result = _execute_task(task, cache)
        cache[key] = result
    result = _execute_task(out, cache)
    if isinstance(out, list):
        result = _lists_to_tuples(result, out)
    return result


def keys_in_tasks(
    keys: Collection[Key],
    tasks: Iterable[Any],
    as_list: bool = False,  # noqa: FBT001, FBT002
):
    """
    Returns the keys in `keys` that are also in `tasks`.

    Examples
    --------
    >>> inc = lambda x: x + 1
    >>> add = lambda x, y: x + y
    >>> dsk = {
    ...     "x": 1,
    ...     "y": (inc, "x"),
    ...     "z": (add, "x", "y"),
    ...     "w": (inc, "z"),
    ...     "a": (add, (inc, "x"), 1),
    ... }

    >>> keys_in_tasks(dsk, ["x", "y", "j"])  # doctest: +SKIP
    {'x', 'y'}
    """
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
    return ret if as_list else set(ret)


def iskey(key: object) -> bool:
    """
    Return True if the given object is a potential dask key; False otherwise.

    The definition of a key in a Dask graph is any str, bytes, int, float, or tuple
    thereof.

    See Also
    --------
    ishashable
    validate_key
    dask.typing.Key
    """
    typ = type(key)
    if typ is tuple:
        return all(iskey(i) for i in cast(tuple, key))
    return typ in {bytes, int, float, str}


def validate_key(key: object) -> None:
    """
    Validate the format of a dask key.

    See Also
    --------
    iskey
    """
    if iskey(key):
        return
    typ = type(key)

    if typ is tuple:
        index = None
        try:
            for index, part in enumerate(cast(tuple, key)):  # noqa: B007
                validate_key(part)
        except TypeError as e:
            raise TypeError(
                f"Composite key contains unexpected key type at {index=} ({key=!r})"
            ) from e
    raise TypeError(f"Unexpected key type {typ} ({key=!r})")


@overload
def get_dependencies(
    dsk: Graph,
    key: Key | None = ...,
    task: Key | NoDefault = ...,
    as_list: Literal[False] = ...,
) -> set[Key]: ...


@overload
def get_dependencies(
    dsk: Graph,
    key: Key | None,
    task: Key | NoDefault,
    as_list: Literal[True],
) -> list[Key]: ...


def get_dependencies(
    dsk: Graph,
    key: Key | None = None,
    task: Key | NoDefault = no_default,
    as_list: bool = False,  # noqa: FBT001, FBT002
) -> set[Key] | list[Key]:
    """
    Get the immediate tasks on which this task depends.

    Examples
    --------
    >>> inc = lambda x: x + 1
    >>> add = lambda x, y: x + y
    >>> dsk = {
    ...     "x": 1,
    ...     "y": (inc, "x"),
    ...     "z": (add, "x", "y"),
    ...     "w": (inc, "z"),
    ...     "a": (add, (inc, "x"), 1),
    ... }

    >>> get_dependencies(dsk, "x")
    set()

    >>> get_dependencies(dsk, "y")
    {'x'}

    >>> get_dependencies(dsk, "z")  # doctest: +SKIP
    {'x', 'y'}

    >>> get_dependencies(dsk, "w")  # Only direct dependencies
    {'z'}

    >>> get_dependencies(dsk, "a")  # Ignore non-keys
    {'x'}

    >>> get_dependencies(dsk, task=(inc, "x"))  # provide tasks directly
    {'x'}
    """
    if key is not None:
        arg = dsk[key]
    elif task is not no_default:
        arg = task
    else:
        raise ValueError("Provide either key or task")

    return keys_in_tasks(dsk, [arg], as_list=as_list)


def flatten(seq, container=list):
    """
    Flatten an arbitrary sequence.

    >>> list(flatten([1]))
    [1]

    >>> list(flatten([[1, 2], [1, 2]]))
    [1, 2, 1, 2]

    >>> list(flatten([[[1], [2]], [[1], [2]]]))
    [1, 2, 1, 2]

    >>> list(flatten(((1, 2), (1, 2))))  # Don't flatten tuples
    [(1, 2), (1, 2)]

    >>> list(flatten((1, 2, [3, 4])))  # support heterogeneous
    [1, 2, 3, 4]
    """
    if isinstance(seq, str):
        yield seq
    else:
        for item in seq:
            if isinstance(item, container):
                yield from flatten(item, container=container)
            else:
                yield item


T_ = TypeVar("T_")


def _reverse_dict(d: Mapping[T_, Iterable[T_]]) -> dict[T_, set[T_]]:
    result: defaultdict[T_, set[T_]] = defaultdict(set)
    _add = set.add
    for k, vals in d.items():
        result[k]
        for val in vals:
            _add(result[val], k)
    return dict(result)


def _toposort(dsk, *, keys=None, returncycle=False, dependencies=None):
    # Stack-based depth-first search traversal.  This is based on Tarjan's
    # method for topological sorting (see wikipedia for pseudocode)
    if keys is None:
        keys = dsk
    elif not isinstance(keys, list):
        keys = [keys]
    if not returncycle:
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

    if dependencies is None:
        dependencies = {k: get_dependencies(dsk, k) for k in dsk}

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

                        if returncycle:
                            return cycle
                        else:
                            cycle = "->".join(str(x) for x in cycle)
                            raise RuntimeError(f"Cycle detected in Dask: {cycle}")
                    next_nodes.append(nxt)

            if next_nodes:
                nodes.extend(next_nodes)
            else:
                # cur has no more descendants to explore, so we're done with it
                if not returncycle:
                    ordered.append(cur)
                completed.add(cur)
                seen.remove(cur)
                nodes.pop()
    if returncycle:
        return []
    return ordered


def toposort(dsk, dependencies=None):
    """Return a list of keys of dask sorted in topological order."""
    return _toposort(dsk, dependencies=dependencies)
