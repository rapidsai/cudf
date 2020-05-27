# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import numpy as np
import pytest

import rmm

import nvstrings
import nvtext


def test_porter_stemmer_measure_without_arguments():
    strs = nvstrings.to_device(
        ["tr", "ee", "tree", "trouble", "troubles", "private"]
    )
    got = nvtext.porter_stemmer_measure(strs)
    expect = [0, 0, 0, 1, 2, 2]

    assert got == expect


def test_porter_stemmer_measure_with_arguments():

    strs = nvstrings.to_device(
        [
            "Tr",
            "Ee",
            "Tree",
            "TrouBle",
            "TROUBLeS",
            "PRIVATE",
            "Country",
            "Country",
        ]
    )
    got = nvtext.porter_stemmer_measure(strs, y_char="y", vowels="AEIOUaeiou")
    expect = [0, 0, 0, 1, 2, 2, 1, 1]

    assert got == expect


def test_porter_stemmer_measure_value_error():
    strs = ["test"]
    # Raise Value Error is not an instance of nvstrings class
    with pytest.raises(ValueError):
        nvtext.porter_stemmer_measure(strs)


def test_vowels():

    strs = nvstrings.to_device(
        ["toys", "syzygy", "buddy", "county", "counties", "private", "", None]
    )
    got = nvtext.is_vowel(strs, 2, y_char="y", vowels="aeiou")
    expect = [False, False, False, True, True, True, False, False]
    assert got == expect

    got = nvtext.is_vowel(strs, 5)
    expect = [False, True, False, True, True, False, False, False]
    assert got == expect

    got = nvtext.is_vowel(strs, -1)
    expect = [False, True, True, True, False, True, False, False]
    assert got == expect

    indices_array = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    darr = rmm.to_device(indices_array)
    got = nvtext.is_vowel(strs, darr.device_ctypes_pointer.value, True)
    expect = [True, False, False, False, True, True, False, False]
    assert got == expect


def test_consonants():

    strs = nvstrings.to_device(
        ["toys", "syzygy", "buddy", "county", "counties", "private", "", None]
    )
    got = nvtext.is_consonant(strs, 2, y_char="y", vowels="aeiou")
    expect = [True, True, True, False, False, False, False, False]
    assert got == expect

    got = nvtext.is_consonant(strs, 5)
    expect = [False, False, False, False, False, True, False, False]
    assert got == expect

    got = nvtext.is_consonant(strs, -2)
    expect = [True, True, True, True, False, True, False, False]
    assert got == expect

    indices_array = np.array([1, 2, 3, 4, -5, 6, 7, 8], dtype=np.int32)
    darr = rmm.to_device(indices_array)
    got = nvtext.is_vowel(strs, darr.device_ctypes_pointer.value, True)
    expect = [True, False, False, False, False, True, False, False]
    assert got == expect
