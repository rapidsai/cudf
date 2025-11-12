# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging

logging.basicConfig(
    filename="cudf_pandas_unit_tests_debug.log", level=logging.INFO
)
logger = logging.getLogger()


class StructuredMessage:
    # https://docs.python.org/3/howto/logging-cookbook.html#implementing-structured-logging
    def __init__(self, debug_type: str, /, **kwargs) -> None:
        self.debug_type = debug_type
        self.kwargs = kwargs

    def __str__(self) -> str:
        log = {"debug_type": self.debug_type}
        return json.dumps({**log, **self.kwargs})


def reprify(arg) -> str:
    """Attempt to return arg's repr for logging."""
    try:
        return repr(arg)
    except Exception:
        return "<REPR FAILED>"


def log_fallback(
    slow_args: tuple, slow_kwargs: dict, exception: Exception
) -> None:
    """Log when a fast call falls back to the slow path."""
    caller = slow_args[0]
    module = getattr(caller, "__module__", "")
    obj_name = getattr(caller, "__qualname__", type(caller).__qualname__)
    if module:
        slow_object = f"{module}.{obj_name}"
    else:
        slow_object = obj_name
    # TODO: Maybe use inspect.signature to map called args and kwargs
    # to their keyword names, but a user calling an API incorrectly would
    # break this.
    caller_args = slow_args[1]
    args_passed = ", ".join((reprify(arg) for arg in caller_args))
    args_types_passed = ", ".join((type(arg).__name__ for arg in caller_args))
    kwargs_passed = {}
    kwargs_types_passed = ""
    if len(slow_args) == 3:
        caller_kwargs = slow_args[2]
        if caller_kwargs:
            fmt_kwargs = ", ".join(
                f"{kwarg}={reprify(value)}"
                for kwarg, value in caller_kwargs.items()
            )
            kwargs_types_passed = ", ".join(
                f"{kwarg}={type(value).__name__}"
                for kwarg, value in caller_kwargs.items()
            )
            args_passed = f"{args_passed}, {fmt_kwargs}"
            kwargs_passed = {
                kwarg: reprify(value) for kwarg, value in caller_kwargs.items()
            }
    message = StructuredMessage(
        "LOG_FAST_FALLBACK",
        failed_call=f"{slow_object}({args_passed})",
        exception=type(exception).__name__,
        exception_message=str(exception),
        slow_object=slow_object,
        args_passed=args_passed,
        kwargs_passed=kwargs_passed,
        args_types_passed=args_types_passed,
        kwargs_types_passed=kwargs_types_passed,
    )
    logger.info(message)
