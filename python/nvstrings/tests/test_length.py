# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import numpy as np
from utils import assert_eq

import rmm

import nvstrings


def test_len():
    strs = nvstrings.to_device(
        [
            "abc",
            "Def",
            None,
            "jLl",
            "mnO",
            "PqR",
            "sTT",
            "dog and cat",
            "accénted",
            "",
            " 1234 ",
            "XYZ",
        ]
    )
    assert len(strs) == 12
    assert strs.len() == [3, 3, None, 3, 3, 3, 3, 11, 8, 0, 6, 3]


def test_size():
    strs = nvstrings.to_device(
        [
            "abc",
            "Def",
            None,
            "jLl",
            "mnO",
            "PqR",
            "sTT",
            "dog and cat",
            "accénted",
            "",
            " 1234 ",
            "XYZ",
        ]
    )
    assert strs.size() == 12


def test_byte_count():
    strs = nvstrings.to_device(
        [
            "abc",
            "Def",
            None,
            "jLl",
            "mnO",
            "PqR",
            "sTT",
            "dog and cat",
            "accénted",
            "",
            " 1234 ",
            "XYZ",
        ]
    )
    assert strs.byte_count() == 47


def test_null_count():
    strs = nvstrings.to_device(
        [
            "abc",
            "Def",
            None,
            "jLl",
            "mnO",
            "PqR",
            "sTT",
            "dog and cat",
            "accénted",
            "",
            " 1234 ",
            "XYZ",
        ]
    )
    assert strs.null_count() == 1


def test_code_points():
    strs = nvstrings.to_device(
        [
            "abc",
            "Def",
            None,
            "jLl",
            "dog and cat",
            "accénted",
            "",
            " 1234 ",
            "XYZ",
        ]
    )
    assert strs.len() == [3, 3, None, 3, 11, 8, 0, 6, 3]
    #
    d_arr = rmm.device_array(37, dtype=np.uint32)
    devmem = d_arr.device_ctypes_pointer.value
    strs.code_points(devmem)
    expected = [
        97,
        98,
        99,
        68,
        101,
        102,
        106,
        76,
        108,
        100,
        111,
        103,
        32,
        97,
        110,
        100,
        32,
        99,
        97,
        116,
        97,
        99,
        99,
        50089,
        110,
        116,
        101,
        100,
        32,
        49,
        50,
        51,
        52,
        32,
        88,
        89,
        90,
    ]
    assert_eq(d_arr.copy_to_host().tolist(), expected)
