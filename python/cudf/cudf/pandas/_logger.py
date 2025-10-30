# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

logger = logging.getLogger(__name__)


def log_fallback(
    slow_args: tuple, slow_kwargs: dict, exception: Exception
) -> None:
    """Log when a fast call falls back to the slow path."""
    caller = slow_args[0]
    module = getattr(caller, "__module__", "")
    slow_object = getattr(caller, "__qualname__", type(caller).__qualname__)
    message = {
        "debug_type": "LOG_FAST_FALLBACK",
        "slow_object": slow_object,
        "module": module,
        "exception_type": type(exception).__name__,
    }
    logger.info(json.dumps(message))
