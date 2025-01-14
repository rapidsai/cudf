# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Base class for IR nodes, and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from typing_extensions import Self


__all__: list[str] = ["Node"]

T = TypeVar("T", bound="Node[Any]")


class Node(Generic[T]):
    """
    An abstract node type.

    Nodes are immutable!

    This contains a (potentially empty) tuple of child nodes,
    along with non-child data. For uniform reconstruction and
    implementation of hashing and equality schemes, child classes need
    to provide a certain amount of metadata when they are defined.
    Specifically, the ``_non_child`` attribute must list, in-order,
    the names of the slots that are passed to the constructor. The
    constructor must take arguments in the order ``(*_non_child,
    *children).``
    """

    __slots__ = ("_hash_value", "_repr_value", "children")
    _hash_value: int
    _repr_value: str
    children: tuple[T, ...]
    _non_child: ClassVar[tuple[str, ...]] = ()

    def _ctor_arguments(self, children: Sequence[T]) -> Sequence[Any | T]:
        return (*(getattr(self, attr) for attr in self._non_child), *children)

    def reconstruct(self, children: Sequence[T]) -> Self:
        """
        Rebuild this node with new children.

        Parameters
        ----------
        children
            New children

        Returns
        -------
        New node with new children. Non-child data is shared with the input.
        """
        return type(self)(*self._ctor_arguments(children))

    def get_hashable(self) -> Hashable:
        """
        Return a hashable object for the node.

        Returns
        -------
        Hashable object.

        Notes
        -----
        This method is used by the :meth:`__hash__` implementation
        (which does caching). If your node type needs special-case
        handling for some of its attributes, override this method, not
        :meth:`__hash__`.
        """
        return (type(self), self._ctor_arguments(self.children))

    def __hash__(self) -> int:
        """
        Hash of an expression with caching.

        See Also
        --------
        get_hashable
        """
        try:
            return self._hash_value
        except AttributeError:
            self._hash_value = hash(self.get_hashable())
            return self._hash_value

    def is_equal(self, other: Self) -> bool:
        """
        Equality of two nodes of equal type.

        Override this in subclasses, rather than :meth:`__eq__`.

        Parameter
        ---------
        other
            object of same type to compare to.

        Notes
        -----
        Since nodes are immutable, this does common subexpression
        elimination when two nodes are determined to be equal.

        :meth:`__eq__` handles the case where the objects being
        compared are not of the same type, so in this method, we only
        need to implement equality of equal types.

        Returns
        -------
        True if the two nodes are equal, false otherwise.
        """
        if self is other:
            return True
        result = self._ctor_arguments(self.children) == other._ctor_arguments(
            other.children
        )
        # Eager CSE for nodes that match.
        if result:
            self.children = other.children
        return result

    def __eq__(self, other: Any) -> bool:
        """
        Equality of expressions.

        See Also
        --------
        is_equal
        """
        if type(self) is not type(other) or hash(self) != hash(other):
            return False
        else:
            return self.is_equal(other)

    def __ne__(self, other: Any) -> bool:
        """Inequality of expressions."""
        return not self.__eq__(other)

    def __repr__(self) -> str:
        """String representation of an expression with caching."""
        try:
            return self._repr_value
        except AttributeError:
            args = ", ".join(f"{arg!r}" for arg in self._ctor_arguments(self.children))
            self._repr_value = f"{type(self).__name__}({args})"
            return self._repr_value
