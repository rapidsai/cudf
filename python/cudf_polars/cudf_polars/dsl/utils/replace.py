# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for replacing nodes in a DAG."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.traversal import CachingVisitor, reuse_if_unchanged

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cudf_polars.typing import GenericTransformer, NodeT

__all__ = ["replace"]


def _replace(node: NodeT, fn: GenericTransformer[NodeT, NodeT]) -> NodeT:
    try:
        return fn.state["replacements"][node]
    except KeyError:
        return reuse_if_unchanged(node, fn)


def replace(nodes: Sequence[NodeT], replacements: Mapping[NodeT, NodeT]) -> list[NodeT]:
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
    mapper: GenericTransformer[NodeT, NodeT] = CachingVisitor(
        _replace, state={"replacements": replacements}
    )
    return [mapper(node) for node in nodes]
