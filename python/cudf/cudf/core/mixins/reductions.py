# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from .mixin_factory import _create_delegating_mixin

Reducible = _create_delegating_mixin(
    "Reducible",
    "Mixin encapsulating reduction operations.",
    "REDUCTION",
    "_reduce",
    {
        "sum",
        "product",
        "min",
        "max",
        "count",
        "any",
        "all",
        "sum_of_squares",
        "mean",
        "var",
        "std",
        "median",
        "argmax",
        "argmin",
        "nunique",
        "nth",
        "collect",
        "unique",
        "prod",
        "idxmin",
        "idxmax",
        "first",
        "last",
    },
)
