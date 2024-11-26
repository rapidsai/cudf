# Copyright (c) 2024, NVIDIA CORPORATION.
import pickle

import pytest

from pylibcudf import DataType
from pylibcudf.types import TypeId


@pytest.fixture(params=list(TypeId))
def dtype(request):
    tid = request.param
    if tid in {TypeId.DECIMAL32, TypeId.DECIMAL64, TypeId.DECIMAL128}:
        scale = 5
    else:
        scale = 0
    return DataType(tid, scale)


def test_reduce(dtype):
    (typ, (tid, scale)) = dtype.__reduce__()
    assert typ is DataType
    assert tid == dtype.id()
    assert scale == dtype.scale()


def test_pickle(dtype):
    assert dtype == pickle.loads(pickle.dumps(dtype))
