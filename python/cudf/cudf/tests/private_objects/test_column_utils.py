# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import pytest

from cudf.core.column.utils import access_columns


class _StubColumn:
    def __init__(self, name, events, fail=False):
        self.name = name
        self.events = events
        self.fail = fail

    @contextmanager
    def access(self, **kwargs):
        self.events.append(f"enter:{self.name}")
        try:
            if self.fail:
                raise RuntimeError("access failed")
            yield self
        finally:
            self.events.append(f"exit:{self.name}")


def test_access_columns_cleanup_on_enter_failure():
    events = []
    first = _StubColumn("first", events)
    second = _StubColumn("second", events, fail=True)

    with pytest.raises(RuntimeError, match="access failed"):
        with access_columns(first, second):
            pytest.fail("access_columns must not enter the with body")

    assert events == [
        "enter:first",
        "enter:second",
        "exit:second",
        "exit:first",
    ]


def test_access_columns_cleanup_on_success():
    events = []
    first = _StubColumn("first", events)
    second = _StubColumn("second", events)

    with access_columns(first, [second]) as accessed:
        assert accessed == (first, second)
        assert events == ["enter:first", "enter:second"]

    assert events == [
        "enter:first",
        "enter:second",
        "exit:second",
        "exit:first",
    ]
