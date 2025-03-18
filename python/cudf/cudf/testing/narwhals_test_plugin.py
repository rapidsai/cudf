# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running narwhals test suite with cudf."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/frame/select_test.py::test_select_duplicates[cudf]": "cuDF doesn't support having multiple columns with same names",
    "tests/translate/from_native_test.py::test_eager_only_sqlframe[False-context0]": "duckdb not installed",
    "tests/translate/from_native_test.py::test_eager_only_sqlframe[True-context1]": "duckdb not installed",
    "tests/translate/from_native_test.py::test_series_only_sqlframe": "duckdb not installed",
}


def pytest_collection_modifyitems(session, config, items) -> None:
    """Mark known failing tests."""
    import pytest

    for item in items:
        if item.nodeid in EXPECTED_FAILURES:
            exp_val = EXPECTED_FAILURES[item.nodeid]
            item.add_marker(pytest.mark.xfail(reason=exp_val))
