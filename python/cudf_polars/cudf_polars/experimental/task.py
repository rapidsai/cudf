# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Task specification."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, Any, TypeAlias

from typing_extensions import Unpack

if TYPE_CHECKING:
    from collections.abc import Sequence


DaskKey: TypeAlias = str | tuple[str, Unpack[tuple[int, ...]]]
DaskTask: TypeAlias = DaskKey | tuple[Callable, Unpack[tuple[Any, ...]]]


class Key:
    """Task key."""

    __slots__ = ("index", "name")
    name: str
    """Key name."""
    index: tuple[int, ...]
    """Key index."""

    def __init__(self, name: str, *index: int) -> None:
        self.name = name
        self.index = index

    def to_dask(self) -> DaskKey:
        """Return Dask-compatible key."""
        if self.index:
            return (self.name, *self.index)
        else:
            return self.name

    def __repr__(self) -> str:
        """String representation of a Key."""
        return f"Key{(self.name, *self.index)}"

    def __hash__(self) -> int:
        """Hash of a Key."""
        return hash((self.name, *self.index))

    def __eq__(self, other: Any) -> bool:
        """Check if Keys are equal."""
        if isinstance(other, Key):
            return self.name == other.name and self.index == other.index
        return False


class Task:
    """
    Compute task.

    Notes
    -----
    See ``Task.execute`` for the required function signature
    of ``Task.function``.
    """

    __slots__ = ("args", "deps", "function")
    function: Callable
    """Callable function."""
    args: Sequence[Any]
    """Positional task arguments."""
    deps: Sequence[Key]
    """Task dependencies."""

    def __init__(
        self, function: Callable, *, args: Sequence[Any] = (), deps: Sequence[Key] = ()
    ) -> None:
        self.function = function
        self.args = args
        self.deps = deps

    def execute(self, cache: MutableMapping) -> Any:
        """
        Execute the task.

        Parameters
        ----------
        cache
            Dictionary containing pre-computed dependencies.

        Returns
        -------
        Output of the computed task.

        Notes
        -----
        The callable function (``Task.function``) will only
        be passed unpacked positional arguments (first ``args``
        and then ``deps``).
        """
        return self.function(*self.args, *(cache[key] for key in self.deps))

    def to_dask(self) -> DaskTask:
        """Return Dask-compatible task."""
        return (self.function, *self.args, *(key.to_dask() for key in self.deps))

    def __repr__(self) -> str:
        """String representation of a Task."""
        return f"Task({self.function}, args={self.args}, deps={self.deps})"


def _task_deps(task_or_key: Key | Task) -> tuple[Key, ...]:
    """
    Return the dependencies of a given Task or Key.

    Notes
    -----
    The only dependency of a Key is the Key itself.
    """
    if isinstance(task_or_key, Key):
        return (task_or_key,)
    return tuple(task_or_key.deps)


TaskGraph: TypeAlias = dict[Key, Key | Task]
DaskTaskGraph: TypeAlias = MutableMapping[Any, Any]
