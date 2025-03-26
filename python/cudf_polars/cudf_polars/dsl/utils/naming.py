# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Name generation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


__all__ = ["unique_names"]


def unique_names(names: Iterable[str]) -> Generator[str, None, None]:
    """
    Generate unique names relative to some known names.

    Parameters
    ----------
    names
        Names we should be unique with respect to.

    Yields
    ------
    Unique names (just using sequence numbers)
    """
    prefix = "_" * max(map(len, names))
    i = 0
    while True:
        yield f"{prefix}{i}"
        i += 1
