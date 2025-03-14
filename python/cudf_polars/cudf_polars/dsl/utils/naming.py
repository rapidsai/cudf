# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Name generation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


__all__ = ["unique_names"]


def unique_names(prefix: str) -> Generator[str, None, None]:
    """
    Generate unique names with a given prefix.

    Parameters
    ----------
    prefix
        Prefix to give to names

    Notes
    -----
    If creating temporary named expressions for a node, create a
    prefix that is as long as the longest key in the schema (doesn't
    matter what it is) and use that.

    Yields
    ------
    Unique names (just using sequence numbers)
    """
    i = 0
    while True:
        yield f"{prefix}{i}"
        i += 1
