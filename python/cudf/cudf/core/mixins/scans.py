# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from .mixin_factory import _create_delegating_mixin

Scannable = _create_delegating_mixin(
    "Scannable",
    "Mixin encapsulating scan operations.",
    "SCAN",
    "_scan",
    {
        "cumsum",
        "cumprod",
        "cummin",
        "cummax",
        "ewma",
    },
)
