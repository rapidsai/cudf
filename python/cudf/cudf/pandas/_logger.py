# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging

# https://docs.python.org/3/howto/logging-cookbook.html#implementing-structured-logging


class StructuredMessage:
    def __init__(self, debug_type: str, /, **kwargs):
        self.debug_type = debug_type
        self.kwargs = kwargs

    def __str__(self):
        log = {"debug_type": self.debug_type}
        return json.dumps({**log, **self.kwargs})


logging.basicConfig(
    filename="cudf_pandas_unit_tests_debug.log", level=logging.INFO
)
logger = logging.getLogger()
