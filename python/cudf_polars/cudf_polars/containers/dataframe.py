# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A dataframe, with some properties."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import cudf._lib.pylibcudf as plc

if TYPE_CHECKING:
    from cudf_polars.containers.column import Column
    from cudf_polars.containers.scalar import Scalar

__all__: list[str] = ["DataFrame"]


class DataFrame:
    """A representation of a dataframe."""

    __slots__ = ("columns", "scalars", "names", "scalar_names", "table")
    columns: list[Column]
    scalars: list[Scalar]
    names: dict[str, int]
    scalar_names: frozenset[str]
    table: plc.Table | None

    def __init__(self, columns: list[Column], scalars: list[Scalar]) -> None:
        self.names = dict(zip((c.name for c in columns), itertools.count(0))) | dict(
            zip((s.name for s in columns), itertools.count(0))
        )
        self.scalar_names = frozenset(s.name for s in scalars)
        self.columns = columns
        self.scalars = scalars
        if len(scalars) == 0:
            self.table = plc.Table(columns)
        else:
            self.table = None

    __iter__ = None

    def __getitem__(self, name: str) -> Column | Scalar:
        """Return column with given name."""
        i = self.names[name]
        if name in self.scalar_names:
            return self.scalars[i]
        else:
            return self.columns[i]
