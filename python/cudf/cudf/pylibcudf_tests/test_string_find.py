# Copyright (c) 2024, NVIDIA CORPORATION.

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc
from cudf._lib.scalar import DeviceScalar


@pytest.fixture(scope="module")
def string_col():
    return pa.array(
        ["AbC", "de", "FGHI", "j", "kLm", "nOPq", None, "RsT", None, "uVw"]
    )


@pytest.mark.parametrize("target", ["a", ""])
def test_find(string_col, target):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.find.find(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value, 0, -1
    )
    expected = pa.Array.from_pandas(
        string_col.to_pandas().str.find(target), type=pa.int32()
    )
    assert_column_eq(got, expected)


def test_find_column(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    target_col = plc.interop.from_arrow(
        pa.array(["A", "d", "F", "j", "k", "n", None, "R", None, "u"])
    )
    expected = pa.array([0, 0, 0, 0, 0, 0, None, 0, None, 0], type=pa.int32())
    got = plc.strings.find.find(plc_col, target_col, 0)
    assert_column_eq(got, expected)


@pytest.mark.parametrize("target", ["a", ""])
def test_rfind(string_col, target):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.find.rfind(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value, 0, -1
    )
    expected = pa.Array.from_pandas(
        string_col.to_pandas().str.rfind(target), type=pa.int32()
    )
    assert_column_eq(got, expected)


@pytest.mark.parametrize("target", ["a", "aB", "Ab", ""])
def test_contains(string_col, target):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.find.contains(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value
    )
    expected = pa.Array.from_pandas(
        string_col.to_pandas().str.contains(target)
    )
    assert_column_eq(got, expected)


def test_contains_column(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    target_col = plc.interop.from_arrow(
        pa.array(["a", "d", "F", "j", "m", "q", None, "R", None, "w"])
    )
    expected = pa.array(
        [False, True, True, True, True, True, None, True, None, True]
    )
    got = plc.strings.find.contains(plc_col, target_col)
    assert_column_eq(got, expected)


@pytest.mark.parametrize("target", ["A", "", "Ab"])
def test_starts_with(string_col, target):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.find.starts_with(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value
    )
    expected = pa.compute.starts_with(string_col, target)
    assert_column_eq(got, expected)


def test_starts_with_column(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    target_col = plc.interop.from_arrow(
        pa.array(["A", "d", "F", "j", "k", "n", None, "R", None, "u"])
    )
    expected = pa.array(
        [True, True, True, True, True, True, None, True, None, True]
    )
    got = plc.strings.find.starts_with(plc_col, target_col)
    assert_column_eq(got, expected)


@pytest.mark.parametrize("target", ["C", "bC", "BC", ""])
def test_ends_with(string_col, target):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.find.ends_with(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value
    )
    expected = pa.compute.ends_with(string_col, target)
    assert_column_eq(got, expected)


def test_ends_with_column(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    target_col = plc.interop.from_arrow(
        pa.array(["C", "e", "I", "j", "m", "q", None, "T", None, "w"])
    )
    expected = pa.array(
        [True, True, True, True, True, True, None, True, None, True]
    )
    got = plc.strings.find.ends_with(plc_col, target_col)
    assert_column_eq(got, expected)
