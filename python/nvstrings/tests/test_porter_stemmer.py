# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings
import nvtext


def test_porter_stemmer_measure():

    # todo not handling Nones here
    strs = nvstrings.to_device(
        ["tr", "ee", "tree", "trouble", "troubles", "private"]
    )
    got = nvtext.porter_stemmer_measure(strs)
    expect = [0, 0, 0, 1, 2, 2]

    assert got == expect
