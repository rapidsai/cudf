# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Traversal and visitor utilities for nodes."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, MutableMapping

    from cudf_polars.dsl.nodebase import Node


def traversal(node: Node) -> Generator[Node, None, None]:
    """
    Pre-order traversal of nodes in an expression.

    Parameters
    ----------
    node
        Root of expression to traverse.

    Yields
    ------
    Unique nodes in the expression, parent before child, children
    in-order from left to right.
    """
    seen = {node}
    lifo = [node]

    while lifo:
        node = lifo.pop()
        yield node
        for child in reversed(node.children):
            if child not in seen:
                seen.add(child)
                lifo.append(child)


U = TypeVar("U", bound=Hashable)
V = TypeVar("V")


def reuse_if_unchanged(e: Node, fn: Callable[[Node], Node]) -> Node:
    """
    Recipe for transforming nodes that returns the old object if unchanged.

    Parameters
    ----------
    e
         Node to recurse on
    fn
         Function to transform children

    Notes
    -----
    This can be used as a generic "base case" handler when
    writing transforms that take nodes and produce new nodes.

    Returns
    -------
    Existing node `e` if transformed children are unchanged, otherwise
    reconstructed node with new children.
    """
    new_children = [fn(c) for c in e.children]
    if all(new == old for new, old in zip(new_children, e.children, strict=True)):
        return e
    return e.reconstruct(new_children)


def make_recursive(fn: Callable[[U, Callable[[U], V]], V]) -> Callable[[U], V]:
    """
    No-op wrapper for recursive visitors.

    Facilitates using visitors that don't need caching but are written
    in the same style.

    Arbitrary immutable state can be attached to the visitor by
    setting properties on the wrapper, since the functions will
    receive the wrapper as an argument.

    Parameters
    ----------
    fn
        Function to transform inputs to outputs. Should take as its
        second argument a callable from input to output.

    Notes
    -----
    All transformation functions *must* be pure.

    Usually, prefer a :class:`CachingVisitor`, but if we know that we
    don't need caching in a transformation and then this no-op
    approach is slightly cheaper.

    Returns
    -------
    Recursive function without caching.

    See Also
    --------
    CachingVisitor
    """

    def rec(node: U) -> V:
        return fn(node, rec)

    return rec


class CachingVisitor(Generic[U, V]):
    """
    Caching wrapper for recursive visitors.

    Facilitates writing visitors where already computed results should
    be cached and reused. The cache is managed automatically, and is
    tied to the lifetime of the wrapper.

    Arbitrary immutable state can be attached to the visitor by
    setting properties on the wrapper, since the functions will
    receive the wrapper as an argument.

    Parameters
    ----------
    fn
        Function to transform inputs to outputs. Should take as its
        second argument the recursive cache manager.

    Notes
    -----
    All transformation functions *must* be pure.

    Returns
    -------
    Recursive function with caching.
    """

    def __init__(self, fn: Callable[[U, Callable[[U], V]], V]) -> None:
        self.fn = fn
        self.cache: MutableMapping[U, V] = {}

    def __call__(self, value: U) -> V:
        """
        Apply the function to a value.

        Parameters
        ----------
        value
            The value to transform.

        Returns
        -------
        A transformed value.
        """
        try:
            return self.cache[value]
        except KeyError:
            return self.cache.setdefault(value, self.fn(value, self))

    if TYPE_CHECKING:
        # Advertise to type-checkers that dynamic attributes are allowed
        def __setattr__(self, name: str, value: Any) -> None: ...  # noqa: D105
        def __getattr__(self, name: str) -> Any: ...  # noqa: D105
