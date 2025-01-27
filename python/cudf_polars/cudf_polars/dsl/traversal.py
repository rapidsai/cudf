# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Traversal and visitor utilities for nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from cudf_polars.typing import U_contra, V_co

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, MutableMapping, Sequence

    from cudf_polars.typing import GenericTransformer, NodeT


__all__: list[str] = [
    "CachingVisitor",
    "make_recursive",
    "reuse_if_unchanged",
    "traversal",
]


def traversal(nodes: Sequence[NodeT]) -> Generator[NodeT, None, None]:
    """
    Pre-order traversal of nodes in an expression.

    Parameters
    ----------
    nodes
        Roots of expressions to traverse.

    Yields
    ------
    Unique nodes in the expressions, parent before child, children
    in-order from left to right.
    """
    seen = set(nodes)
    lifo = list(nodes)

    while lifo:
        node = lifo.pop()
        yield node
        for child in reversed(node.children):
            if child not in seen:
                seen.add(child)
                lifo.append(child)


def reuse_if_unchanged(node: NodeT, fn: GenericTransformer[NodeT, NodeT]) -> NodeT:
    """
    Recipe for transforming nodes that returns the old object if unchanged.

    Parameters
    ----------
    node
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
    new_children = [fn(c) for c in node.children]
    if all(new == old for new, old in zip(new_children, node.children, strict=True)):
        return node
    return node.reconstruct(new_children)


def make_recursive(
    fn: Callable[[U_contra, GenericTransformer[U_contra, V_co]], V_co],
    *,
    state: Mapping[str, Any] | None = None,
) -> GenericTransformer[U_contra, V_co]:
    """
    No-op wrapper for recursive visitors.

    Facilitates using visitors that don't need caching but are written
    in the same style.

    Parameters
    ----------
    fn
        Function to transform inputs to outputs. Should take as its
        second argument a callable from input to output.
    state
        Arbitrary *immutable* state that should be accessible to the
        visitor through the `state` property.

    Notes
    -----
    All transformation functions *must* be free of side-effects.

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

    def rec(node: U_contra) -> V_co:
        return fn(node, rec)  # type: ignore[arg-type]

    rec.state = state if state is not None else {}  # type: ignore[attr-defined]
    return rec  # type: ignore[return-value]


class CachingVisitor(Generic[U_contra, V_co]):
    """
    Caching wrapper for recursive visitors.

    Facilitates writing visitors where already computed results should
    be cached and reused. The cache is managed automatically, and is
    tied to the lifetime of the wrapper.

    Parameters
    ----------
    fn
        Function to transform inputs to outputs. Should take as its
        second argument the recursive cache manager.
    state
        Arbitrary *immutable* state that should be accessible to the
        visitor through the `state` property.

    Notes
    -----
    All transformation functions *must* be free of side-effects.

    Returns
    -------
    Recursive function with caching.
    """

    def __init__(
        self,
        fn: Callable[[U_contra, GenericTransformer[U_contra, V_co]], V_co],
        *,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        self.cache: MutableMapping[U_contra, V_co] = {}
        self.state = state if state is not None else {}

    def __call__(self, value: U_contra) -> V_co:
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
