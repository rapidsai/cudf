# Copyright (c) 2022-2024, NVIDIA CORPORATION.

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
    },
)
