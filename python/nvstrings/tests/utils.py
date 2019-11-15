# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np

import nvstrings


def methodcaller(name, *args, **kwargs):
    def caller(obj):
        return getattr(obj, name)(*args, **kwargs)

    return caller


def assert_eq(arr1, arr2):
    if isinstance(arr1, nvstrings.nvstrings):
        arr1 = arr1.to_host()

    if isinstance(arr2, nvstrings.nvstrings):
        arr2 = arr2.to_host()

    assert np.array_equiv(arr1, arr2)
