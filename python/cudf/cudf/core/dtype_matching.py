# Copyright (c) 2023, NVIDIA CORPORATION.
from __future__ import annotations

import functools
import itertools
from typing import Dict, List, Set, Tuple, TypeVar, cast

import numpy as np
from pandas.api.types import is_numeric_dtype

import cudf.core.column as column
from cudf.core.dtypes import is_categorical_dtype

T = TypeVar("T")


def toposort(graph: Dict[T, Set[T]]) -> List[T]:
    """
    Return nodes in a DAG in a (non-unique) topological order.

    The order may not be unique.

    Raises RuntimeError if the provided graph is not a DAG.
    """
    # Kahn's algorithm, https://dl.acm.org/doi/10.1145/368996.369025
    # Copy because we'll modify the graph in place.
    graph = {k: set(v) for k, v in graph.items()}
    invgraph: Dict[T, Set[T]] = {k: set() for k in graph}
    for k, v in graph.items():
        for v_ in v:
            invgraph[v_].add(k)
    order = []
    predecessors = set(k for k, v in invgraph.items() if not v)
    while predecessors:
        order.append(n := predecessors.pop())
        edges = graph[n]
        while edges:
            invgraph[m := edges.pop()].remove(n)
            if not invgraph[m]:
                predecessors.add(m)
    if any(v for v in graph.values()):
        raise RuntimeError("Provided lattice structure is not a DAG")
    return order


def ancestors(graph: Dict[T, Set[T]]) -> Dict[T, Set[T]]:
    """
    Return the graph of the ancestors of a DAG

    Arrows in the graph should point from child to parent.

    Raises RuntimeError if the provided graph is not a DAG.
    """
    graph = {k: set(v) for k, v in graph.items()}
    order = toposort(graph)
    for k in reversed(order):
        graph[k] |= set(
            itertools.chain.from_iterable(graph[v_] for v_ in graph[k])
        )
    return graph


def least_common_ancestors(
    a: T, b: T, *, ancestors: Dict[T, Set[T]]
) -> Set[T]:
    r"""
    Return the least common ancestors of a and b

    Arrows in the graph should point from child to parent

    For example, given the graph

    C  D
    ^  ^
    |\/|
    |/\|
    A  B

    LCA(A, B) = (C, D)
    LCA(A, C) = (C)
    LCA(A, D) = (D)
    LCA(B, C) = (C)
    LCA(B, D) = (D)
    LCA(C, D) = ()
    """
    # This is not very sophisticated but the graphs are tiny.
    # It's basically the algorithm described in Section 4.3 of Bender,
    # Farach-Colton, Pemmasani, Skiena, and Sumazin, J. Algorithms
    # (2005) https://doi.org/10.1016/j.jalgor.2005.08.001

    # Candidate subgraph is intersection of the closure of the nodes
    # closure is ancestors + self.
    subnodes = (ancestors[a] | {a}) & (ancestors[b] | {b})

    # least common ancestors are all nodes in the subgraph that are
    # roots (i.e. have no edges pointing in to them)
    return subnodes - set(
        itertools.chain.from_iterable(ancestors[node] for node in subnodes)
    )


def promotion_table(graph: Dict[T, Set[T]]) -> Dict[Tuple[T, T], T]:
    """
    Return a type promotion table for all pairs of types in the
    provided partial semi-lattice.

    Raises RuntimeError if the input is not a DAG

    Raises ValueError if any pair does not have a unique least upper
    bound.

    The promotion for each pair in the input is the unique least upper
    bound of the pair in the input. The absence of a pair in the
    return value means that no unique upper bound exists, so there is
    no safe promotion route for that pair.

    Arrows in the graph should point from child to parent.
    """
    successors = ancestors(graph)
    table = dict()
    for pair in itertools.product(graph, graph):
        lca = least_common_ancestors(*pair, ancestors=successors)
        if len(lca) > 1:
            # Malformed input
            raise ValueError("Provided DAG is not a partial semi-lattice")
        try:
            # Is there a unique least upper bound?
            (table[pair],) = lca
        except ValueError:
            # No, no safe promotion route
            pass
    return table


@functools.cache
def numeric_promotions() -> Dict[Tuple[np.dtype, np.dtype], np.dtype]:
    """
    Return a type promotion table for all pairs of non-decimal numeric types.

    If a pair exists as a key in the return value, the value is the
    safe type to promote to. If a pair does not exist, there is no
    safe promotion route.
    """
    # Direct promotion routes for integer types
    # +----+    +-----+    +-----+    +-----+
    # | i8 |--->| i16 |--->| i32 |--->| i64 |
    # +----+    +--^--+    +--^--+    +--^--+
    #             /          /          /
    #         /---'      /---'      /---'
    #     /---'      /---'      /---'
    # +--+-+    +---+-+    +---+-+    +-----+
    # | u8 |--->| u16 |--->| u32 |--->| u64 |
    # +----+    +-----+    +-----+    +-----+
    int_lattice: Dict[np.dtype, Set[np.dtype]] = {
        np.dtype("uint8"): {np.dtype("uint16"), np.dtype("int16")},
        np.dtype("int8"): {np.dtype("int16")},
        np.dtype("uint16"): {np.dtype("uint32"), np.dtype("int32")},
        np.dtype("int16"): {np.dtype("int32")},
        np.dtype("uint32"): {np.dtype("uint64"), np.dtype("int64")},
        np.dtype("int32"): {np.dtype("int64")},
        np.dtype("uint64"): set(),
        np.dtype("int64"): set(),
    }
    # Direct promotion routes for float types
    # +-----+    +-----+    +-----+
    # | f16 |--->| f32 |--->| f64 |
    # +-----+    +-----+    +-----+
    float_lattice: Dict[np.dtype, Set[np.dtype]] = {
        np.dtype("float16"): {np.dtype("float32")},
        np.dtype("float32"): {np.dtype("float64")},
        np.dtype("float64"): set(),
    }
    return {**promotion_table(int_lattice), **promotion_table(float_lattice)}


def common_numeric_type(ltype: np.dtype, rtype: np.dtype) -> np.dtype | None:
    """Return the common type between ltype and rtype or None.

    If there exists a common type it is safe to losslessly promote the
    provided pair of dtype to, it is returned. If there is no safe
    promotion route, None is returned, and it is up to the caller to
    decide how to proceed.
    """
    return numeric_promotions().get((ltype, rtype), None)


def match_join_types(
    left: column.ColumnBase, right: column.ColumnBase
) -> Tuple[column.ColumnBase, column.ColumnBase]:
    """Given a pair of columns to join, return a new pair with
    matching dtypes

    Parameters
    ----------
    left
        Left column to join on
    right
        Right column to join on

    Returns
    -------
    tuple
        Pair of, possibly type-promoted, left and right columns with
        matching dtype

    Raises
    ------
    ValueError
        If there exists no safe promotion rule for the pair of
        columns.

    Notes
    -----
    Non-decimal numeric types are promoted according to the table provided by
    :func:`numeric_promotions` which only allows safe promotions and
    never promotes between numeric kinds. If exactly one column is categorical,
    it is decategorized and promotion continues with the decategorized
    column.

    All other dtypes must match exactly, so there is no automatic
    promotion between (for example) decimal columns of different precision.
    """
    ltype = left.dtype
    rtype = right.dtype

    if ltype == rtype:
        return left, right
    left_is_cat = is_categorical_dtype(left.dtype)
    right_is_cat = is_categorical_dtype(right.dtype)

    if left_is_cat and right_is_cat:
        raise ValueError("Cannot merge on non-matching categorical types")
    elif left_is_cat:
        return match_join_types(
            cast(column.CategoricalColumn, left)._get_decategorized_column(),
            right,
        )
    elif right_is_cat:
        return match_join_types(
            left,
            cast(column.CategoricalColumn, right)._get_decategorized_column(),
        )
    else:
        pass
    if is_numeric_dtype(ltype) and is_numeric_dtype(rtype):
        common_type = common_numeric_type(
            cast(np.dtype, ltype), cast(np.dtype, rtype)
        )
        if common_type is None:
            raise ValueError(
                f"Cannot safely promote numeric pair {ltype} and {rtype}. "
                "To perform the merge, manually cast the merge keys to "
                "an appropriate common type first."
            )
        return left.astype(common_type), right.astype(common_type)

    raise ValueError(
        f"Cannot merge on non-matching key types {ltype} and {rtype}"
    )
