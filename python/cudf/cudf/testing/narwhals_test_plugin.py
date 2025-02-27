# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running narwhals test suite with cudf."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/frame/select_test.py::test_select_duplicates[cudf]": "cuDF doesn't support having multiple columns with same names",
}


def pytest_collection_modifyitems(session, config, items) -> None:
    """Mark known failing tests."""
    import pytest

    for item in items:
        if item.nodeid in EXPECTED_FAILURES:
            exp_val = EXPECTED_FAILURES[item.nodeid]
            item.add_marker(pytest.mark.xfail(reason=exp_val))
