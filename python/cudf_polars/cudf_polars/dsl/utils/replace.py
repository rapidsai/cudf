# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for replacing nodes in a DAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from cudf_polars.dsl.traversal import CachingVisitor
from cudf_polars.typing import NodeT, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cudf_polars.typing import GenericTransformer

__all__ = ["replace"]


class State(Generic[NodeT], TypedDict):
    """
    State used when replacing nodes in expressions.

    Parameters
    ----------
    replacements
        Mapping from nodes to be replaced to their replacements.
        This state is generic over the type of these nodes.
    """

    replacements: Mapping[NodeT, NodeT]


def _replace(
    node: NodeT, fn: GenericTransformer[NodeT, NodeT, State]
) -> NodeT:  # pragma: no cover
    # only called in Window node dispatch function
    # in translate.py, which also skips code coverage
    # See the TODO there for more details.
    try:
        return fn.state["replacements"][node]
    except KeyError:
        # replacement must propagate when children are rebuilt,
        # even if child __eq__ intentionally ignores children (e.g. Cache).
        # So we use identity, not equality like reuse_if_unchanged, to decide unchanged.
        new_children = [fn(c) for c in node.children]
        if all(
            new is old for new, old in zip(new_children, node.children, strict=True)
        ):
            return node
        return node.reconstruct(new_children)


def replace(
    nodes: Sequence[NodeT], replacements: Mapping[NodeT, NodeT]
) -> list[NodeT]:  # pragma: no cover
    """
    Replace nodes in expressions.

    Parameters
    ----------
    nodes
        Sequence of nodes to perform replacements in.
    replacements
        Mapping from nodes to be replaced to their replacements.

    Returns
    -------
    list
        Of nodes with replacements performed.
    """
    mapper: GenericTransformer[NodeT, NodeT, State] = CachingVisitor(
        _replace, state={"replacements": replacements}
    )
    return [mapper(node) for node in nodes]
