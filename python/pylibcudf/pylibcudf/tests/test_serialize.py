# Copyright (c) 2024, NVIDIA CORPORATION.
import pickle

import pytest

from pylibcudf import DataType
from pylibcudf.types import TypeId


@pytest.mark.parametrize("tid", TypeId)
def test_reduce(tid):
    dt = DataType(tid, 0)
    reduced_dt = dt.__reduce__()
    assert reduced_dt[0] is DataType
    assert reduced_dt[1][0] == tid
    assert reduced_dt[1][1] == 0


@pytest.mark.parametrize(
    "tid", [TypeId.DECIMAL32, TypeId.DECIMAL64, TypeId.DECIMAL128]
)
def test_reduce_decimal(tid):
    dt = DataType(tid, 10)
    reduced_dt = dt.__reduce__()
    assert reduced_dt[0] is DataType
    assert reduced_dt[1][0] == tid
    assert reduced_dt[1][1] == 10


@pytest.mark.parametrize("tid", TypeId)
def test_pickle(tid):
    dt = DataType(tid, 0)
    serialized = pickle.dumps(dt)
    dt_got = pickle.loads(serialized)
    assert dt_got == dt


@pytest.mark.parametrize(
    "tid", [TypeId.DECIMAL32, TypeId.DECIMAL64, TypeId.DECIMAL128]
)
def test_pickle_decimal(tid):
    dt = DataType(tid, 0)
    serialized = pickle.dumps(dt)
    dt_got = pickle.loads(serialized)
    assert dt_got == dt
