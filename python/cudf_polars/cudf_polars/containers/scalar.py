# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A scalar, with some properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cudf._lib.pylibcudf as plc

__all__: list[str] = ["Scalar"]


class Scalar:
    """A scalar, and a name."""

    __slots__ = ("obj", "name")
    obj: plc.Scalar

    def __init__(self, scalar: plc.Scalar):
        self.obj = scalar
