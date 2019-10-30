# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import pytest

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
    # Raise Value Error is not a instance of nvstrings class
    with pytest.raises(ValueError):
        nvtext.porter_stemmer_measure(strs)
