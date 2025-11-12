# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np
import pytest

import rmm

import pylibcudf as plc

DTYPES = [
    "u1",
    "i2",
    "f4",
    "f8",
    "f16",
]
SIZES = [
    0,
    1,
    1000,
    1024,
    10000,
]


@pytest.fixture(params=tuple(itertools.product(SIZES, DTYPES)), ids=repr)
def np_array(request):
    size, dtype = request.param
    return np.empty((size,), dtype=dtype)


def test_cuda_array_interface(np_array):
    buf = rmm.DeviceBuffer(
        ptr=np_array.__array_interface__["data"][0], size=np_array.nbytes
    )
    gpumemview = plc.gpumemoryview(buf)

    np_array_view = np_array.view("u1")

    ai = np_array_view.__array_interface__
    cai = gpumemview.__cuda_array_interface__
    assert cai["shape"] == ai["shape"]
    assert cai["strides"] == ai["strides"]
    assert cai["typestr"] == ai["typestr"]


def test_len(np_array):
    buf = rmm.DeviceBuffer(
        ptr=np_array.__array_interface__["data"][0], size=np_array.nbytes
    )
    gpumemview = plc.gpumemoryview(buf)

    np_array_view = np_array.view("u1")

    assert len(gpumemview) == len(np_array_view)
    assert gpumemview.nbytes == np_array.nbytes
