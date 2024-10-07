# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generic expression rewriting utilities."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from functools import singledispatch
from typing import TYPE_CHECKING, Generic, TypeVar

from cudf_polars.dsl.expr import Col, Expr

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableMapping


@singledispatch
def _rename(expr: Expr, self: Callable[[Expr], Expr]) -> Expr:
    raise AssertionError(f"Unhandled type {type(expr)}")


def reuse(expr: Expr, self: Callable[[Expr], Expr]) -> Expr:
    """
    Reuse an expression if transformed children are unchanged.

    Parameters
    ----------
    expr
        Expression to transform.
    self
        callable object that transforms expressions.

    Returns
    -------
    Transformed expression, or input expression if unchanged.
    """
    children = tuple(self(c) for c in expr.children)
    if all(
        c_new is c_old for c_new, c_old in zip(children, expr.children, strict=True)
    ):
        return expr
    return type(expr)(*expr._ctor_arguments(children))


_rename.register(Expr)(reuse)


@_rename.register
def _(expr: Col, self: Callable[[Expr], Expr]) -> Expr:
    return type(expr)(
        expr.dtype,
        self.renamer(expr.name),  # type: ignore[attr-defined]
    )


T = TypeVar("T")
U = TypeVar("U", bound=Hashable)


class Memoizer(Generic[U, T]):
    """
    Memoizing recursive visitor.

    This class facilitates writing visitors where a cache of already
    computed results is managed automatically.

    Parameters
    ----------
    fn
        Callable that takes (hashable) objects of type `U` and returns
        objects of type `T`. It should take, as a second argument, the
        recursive visitor object.
    """

    def __init__(self, fn: Callable[[U, Memoizer[U, T]], T]):
        self.cache: MutableMapping[U, T] = {}
        self.fn = fn

    def __call__(self, node: U) -> T:
        "Caching call to transform node."
        try:
            return self.cache[node]
        except KeyError:
            return self.cache.setdefault(node, self.fn(node, self))


def rename(expr: Iterable[Expr], renamer: Callable[[str], str]) -> list[Expr]:
    """
    Rename column references in expressions.

    Parameters
    ----------
    expr
        Expressions in which to rename columns.
    renamer
        Function to produce a new name given an existing one.

    Returns
    -------
    List of renamed expressions.
    """
    mapper: Memoizer[Expr, Expr] = Memoizer(_rename)
    mapper.renamer = renamer  # type: ignore
    return [mapper(e) for e in expr]
