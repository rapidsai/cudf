# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import numpy as np

import nvstrings

from utils import assert_eq
from librmm_cffi import librmm as rmm


def test_from_offsets():
    values = np.array([97, 112, 112, 108, 101], dtype=np.int8)
    offsets = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    s = nvstrings.from_offsets(values, offsets, 5)
    expected = ['a', 'p', 'p', 'l', 'e']
    assert_eq(s, expected)

    values = np.array([97, 112, 112, 108, 101, 112, 101, 97, 114],
                      dtype=np.int8)
    offsets = np.array([0, 5, 5, 9], dtype=np.int32)
    s = nvstrings.from_offsets(values, offsets, 3)
    expected = ['apple', '', 'pear']
    assert_eq(s, expected)


def test_from_offsets_with_bitmask():
    values = np.array([97, 112, 112, 108, 101], dtype=np.int8)
    offsets = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    bitmask = np.array([29], dtype=np.int8)
    s = nvstrings.from_offsets(values, offsets, 5, bitmask, 1)
    expected = ['a', None, 'p', 'l', 'e']
    assert_eq(s, expected)


def test_from_offsets_ctypes_data():
    values = np.array([97, 112, 112, 108, 101, 112, 101, 97, 114],
                      dtype=np.int8)
    offsets = np.array([0, 5, 5, 9], dtype=np.int32)
    bitmask = np.array([5], dtype=np.int8)
    s = nvstrings.from_offsets(values.ctypes.data, offsets.ctypes.data, 3,
                               bitmask.ctypes.data, 1)
    expected = ['apple', None, 'pear']
    assert_eq(s, expected)


def test_from_offsets_dev_data():
    values = np.array([97, 112, 112, 108, 101, 112, 101, 97, 114],
                      dtype=np.int8)
    offsets = np.array([0, 5, 5, 9], dtype=np.int32)
    bitmask = np.array([5], dtype=np.int8)
    values = rmm.to_device(values)
    offsets = rmm.to_device(offsets)
    bitmask = rmm.to_device(bitmask)
    s = nvstrings.from_offsets(values.device_ctypes_pointer.value,
                               offsets.device_ctypes_pointer.value, 3,
                               bitmask.device_ctypes_pointer.value, 1,
                               True)
    expected = ['apple', None, 'pear']
    assert_eq(s, expected)


def test_to_offsets():
    s = nvstrings.to_device(['a', 'p', 'p', 'l', 'e'])
    values = np.empty(s.size(), dtype=np.int8)
    offsets = np.empty(s.size() + 1, dtype=np.int32)
    nulls = np.empty(int(s.size() / 8) + 1, dtype=np.int8)
    s.to_offsets(values, offsets, nulls)

    expected_values = [97, 112, 112, 108, 101]
    expected_offsets = [0, 1, 2, 3, 4, 5]
    expected_nulls = [31]

    assert np.array_equal(values, expected_values)
    assert np.array_equal(offsets, expected_offsets)
    assert np.array_equal(nulls, expected_nulls)
