# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import rmm

import pylibcudf as plc
from pylibcudf.span import Span, is_span


class MockSpan:
    """Mock object that satisfies Span protocol."""

    def __init__(self, ptr: int, size: int, element_type: type):
        self._ptr = ptr
        self._size = size
        self._element_type = element_type

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def size(self) -> int:
        return self._size

    @property
    def element_type(self) -> type:
        return self._element_type


class MockSpanMissingPtr:
    """Mock object missing ptr attribute."""

    @property
    def size(self) -> int:
        return 100

    @property
    def element_type(self) -> type:
        return int


class MockSpanMissingSize:
    """Mock object missing size attribute."""

    @property
    def ptr(self) -> int:
        return 0

    @property
    def element_type(self) -> type:
        return int


class MockSpanMissingElementType:
    """Mock object missing element_type attribute."""

    @property
    def ptr(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 100


def test_is_span_with_valid_mock():
    """Test is_span() returns True for valid mock Span object."""
    mock = MockSpan(ptr=12345, size=100, element_type=int)
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


def test_is_span_missing_element_type():
    """Test is_span() returns False for object missing element_type."""
    mock = MockSpanMissingElementType()
    assert not is_span(mock)


def test_isinstance_span_with_runtime_checkable():
    """Test isinstance() with runtime_checkable Span protocol."""
    mock = MockSpan(ptr=12345, size=100, element_type=int)
    assert isinstance(mock, Span)


def test_isinstance_span_rejects_incomplete():
    """Test isinstance() rejects objects missing required attributes."""
    assert not isinstance(MockSpanMissingPtr(), Span)
    assert not isinstance(MockSpanMissingSize(), Span)
    assert not isinstance(MockSpanMissingElementType(), Span)


def test_span_element_type_is_int():
    """Test that element_type returns int (representing char)."""
    buf = rmm.DeviceBuffer(size=100)
    gmv = plc.gpumemoryview(buf)
    assert gmv.element_type == int


def test_mock_span_attributes():
    """Test that mock Span object has correct attributes."""
    mock = MockSpan(ptr=0xDEADBEEF, size=256, element_type=int)
    assert mock.ptr == 0xDEADBEEF
    assert mock.size == 256
    assert mock.element_type == int
