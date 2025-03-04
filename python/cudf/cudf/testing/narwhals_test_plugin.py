# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running narwhals test suite with cudf."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/frame/select_test.py::test_select_duplicates[cudf]": "cuDF doesn't support having multiple columns with same names",
    "tests/expr_and_series/over_test.py::test_over_cummin[cudf]": "NotImplementedError: Passing kwargs to func is currently not supported",
    "tests/expr_and_series/over_test.py::test_over_anonymous_cumulative[cudf]": "NotImplementedError: Passing kwargs to func is currently not supported",
    "tests/expr_and_series/over_test.py::test_over_cummax[cudf]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/over_test.py::test_over_cumprod[cudf]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/over_test.py::test_over_cum_reverse[cudf-cum_max-expected_b0]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/over_test.py::test_over_shift[cudf]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/over_test.py::test_over_cumcount[cudf]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/over_test.py::test_over_cum_reverse[cudf-cum_prod-expected_b4]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/over_test.py::test_over_diff[cudf]": "AttributeError: type object 'Aggregation' has no attribute 'diff'",
    "tests/expr_and_series/over_test.py::test_over_cum_reverse[cudf-cum_count-expected_b3]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/over_test.py::test_over_cum_reverse[cudf-cum_sum-expected_b2]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/over_test.py::test_over_cum_reverse[cudf-cum_min-expected_b1]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/over_test.py::test_over_cumsum[cudf]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/rank_test.py::test_rank_expr_in_over_context[cudf-ordinal]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/rank_test.py::test_rank_expr_in_over_context[cudf-dense]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/rank_test.py::test_rank_expr_in_over_context[cudf-min]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/rank_test.py::test_rank_expr_in_over_context[cudf-average]": "NotImplementedError: Passing kwargs to func is currently not supported.",
    "tests/expr_and_series/rank_test.py::test_rank_expr_in_over_context[cudf-max]": "NotImplementedError: Passing kwargs to func is currently not supported.",
}


def pytest_collection_modifyitems(session, config, items) -> None:
    """Mark known failing tests."""
    import pytest

    for item in items:
        if item.nodeid in EXPECTED_FAILURES:
            exp_val = EXPECTED_FAILURES[item.nodeid]
            item.add_marker(pytest.mark.xfail(reason=exp_val))
