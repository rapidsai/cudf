# Copyright (c) 2024, NVIDIA CORPORATION.
import pickle

import pytest

from pylibcudf import DataType
from pylibcudf.libcudf.types import type_id


@pytest.mark.parametrize("tid", [t for t in type_id])
def test_reduce(tid):
    dt = DataType(tid, 0)
    reduced_dt = dt.__reduce__()
    assert reduced_dt[0] is DataType
    assert reduced_dt[1][0] == tid
    assert reduced_dt[1][1] == 0


@pytest.mark.parametrize(
    "tid", [type_id.DECIMAL32, type_id.DECIMAL64, type_id.DECIMAL128]
)
def test_reduce_decimal(tid):
    dt = DataType(tid, 10)
    reduced_dt = dt.__reduce__()
    assert reduced_dt[0] is DataType
    assert reduced_dt[1][0] == tid
    assert reduced_dt[1][1] == 10


@pytest.mark.parametrize("tid", [t for t in type_id])
def test_pickle(tid):
    dt = DataType(tid, 0)
    serialized = pickle.dumps(dt)
    dt_got = pickle.loads(serialized)
    assert dt_got == dt


@pytest.mark.parametrize(
    "tid", [type_id.DECIMAL32, type_id.DECIMAL64, type_id.DECIMAL128]
)
def test_pickle_decimal(tid):
    dt = DataType(tid, 0)
    serialized = pickle.dumps(dt)
    dt_got = pickle.loads(serialized)
    assert dt_got == dt
