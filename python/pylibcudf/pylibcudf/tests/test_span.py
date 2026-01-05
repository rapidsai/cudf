# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import rmm

import pylibcudf as plc
from pylibcudf.span import Span, is_span


class MockSpan:
    """Mock object that satisfies Span protocol."""

    def __init__(self, ptr: int, size: int):
        self._ptr = ptr
        self._size = size

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def size(self) -> int:
        return self._size


class MockSpanMissingPtr:
    """Mock object missing ptr attribute."""

    @property
    def size(self) -> int:
        return 100


class MockSpanMissingSize:
    """Mock object missing size attribute."""

    @property
    def ptr(self) -> int:
        return 0


def test_is_span_with_valid_mock():
    """Test is_span() returns True for valid mock Span object."""
    mock = MockSpan(ptr=12345, size=100)
    assert is_span(mock)


def test_is_span_with_gpumemoryview():
    """Test is_span() returns True for gpumemoryview."""
    buf = rmm.DeviceBuffer(size=100)
    gmv = plc.gpumemoryview(buf)
    assert is_span(gmv)


def test_is_span_with_none():
    """Test is_span() returns False for None."""
    assert not is_span(None)


def test_is_span_missing_ptr():
    """Test is_span() returns False for object missing ptr."""
    mock = MockSpanMissingPtr()
    assert not is_span(mock)


def test_is_span_missing_size():
    """Test is_span() returns False for object missing size."""
    mock = MockSpanMissingSize()
    assert not is_span(mock)


def test_isinstance_span_with_runtime_checkable():
    """Test isinstance() with runtime_checkable Span protocol."""
    mock = MockSpan(ptr=12345, size=100)
    assert isinstance(mock, Span)


def test_isinstance_span_rejects_incomplete():
    """Test isinstance() rejects objects missing required attributes."""
    assert not isinstance(MockSpanMissingPtr(), Span)
    assert not isinstance(MockSpanMissingSize(), Span)


def test_mock_span_attributes():
    """Test that mock Span object has correct attributes."""
    mock = MockSpan(ptr=0xDEADBEEF, size=256)
    assert mock.ptr == 0xDEADBEEF
    assert mock.size == 256
