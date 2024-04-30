# Copyright (c) 2024, NVIDIA CORPORATION.

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc
from cudf._lib.scalar import DeviceScalar


@pytest.fixture(scope="module")
def plc_col():
    data = pa.array(
        ["AbC", "de", "FGHI", "j", "kLm", "nOPq", None, "RsT", None, "uVw"]
    )
    return plc.interop.from_arrow(data)


@pytest.mark.parametrize("target", ["a", ""])
def test_find(plc_col, target):
    got = plc.strings.find.find(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value, 0, -1
    )

    expected = pa.array(
        [
            elem.find(target) if elem is not None else None
            for elem in plc.interop.to_arrow(plc_col).to_pylist()
        ],
        type=pa.int32(),
    )

    assert_column_eq(got, expected)


def test_find_column(plc_col):
    target_col = plc.interop.from_arrow(
        pa.array(["A", "d", "F", "j", "k", "n", None, "R", None, "u"])
    )

    expected = pa.array(
        [
            elem.find(target) if elem is not None else None
            for elem, target in zip(
                plc.interop.to_arrow(plc_col).to_pylist(),
                plc.interop.to_arrow(target_col).to_pylist(),
            )
        ],
        type=pa.int32(),
    )

    got = plc.strings.find.find(plc_col, target_col, 0)
    assert_column_eq(got, expected)


@pytest.mark.parametrize("target", ["a", ""])
def test_rfind(plc_col, target):
    got = plc.strings.find.rfind(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value, 0, -1
    )

    expected = pa.array(
        [
            elem.rfind(target) if elem is not None else None
            for elem in plc.interop.to_arrow(plc_col).to_pylist()
        ],
        type=pa.int32(),
    )

    assert_column_eq(got, expected)


@pytest.mark.parametrize("target", ["a", "aB", "Ab", ""])
def test_contains(plc_col, target):
    got = plc.strings.find.contains(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value
    )
    expected = pa.array(
        [
            target in elem if elem is not None else None
            for elem in plc.interop.to_arrow(plc_col).to_pylist()
        ],
        type=pa.bool_(),
    )

    assert_column_eq(got, expected)


def test_contains_column(plc_col):
    target_col = plc.interop.from_arrow(
        pa.array(["a", "d", "F", "j", "m", "q", None, "R", None, "w"])
    )

    expected = pa.array(
        [
            target in elem if elem is not None else None
            for elem, target in zip(
                plc.interop.to_arrow(plc_col).to_pylist(),
                plc.interop.to_arrow(target_col).to_pylist(),
            )
        ],
        type=pa.bool_(),
    )

    got = plc.strings.find.contains(plc_col, target_col)
    assert_column_eq(got, expected)


@pytest.mark.parametrize("target", ["A", "", "Ab"])
def test_starts_with(plc_col, target):
    got = plc.strings.find.starts_with(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value
    )
    expected = pa.compute.starts_with(plc.interop.to_arrow(plc_col), target)
    assert_column_eq(got, expected)


def test_starts_with_column(plc_col):
    target_col = plc.interop.from_arrow(
        pa.array(["A", "d", "F", "j", "k", "n", None, "R", None, "u"])
    )

    expected = pa.array(
        [
            elem.startswith(target) if elem is not None else None
            for elem, target in zip(
                plc.interop.to_arrow(plc_col).to_pylist(),
                plc.interop.to_arrow(target_col).to_pylist(),
            )
        ],
        type=pa.bool_(),
    )

    got = plc.strings.find.starts_with(plc_col, target_col)
    assert_column_eq(got, expected)


@pytest.mark.parametrize("target", ["C", "bC", "BC", ""])
def test_ends_with(plc_col, target):
    got = plc.strings.find.ends_with(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value
    )
    expected = pa.compute.ends_with(plc.interop.to_arrow(plc_col), target)
    assert_column_eq(got, expected)


def test_ends_with_column(plc_col):
    target_col = plc.interop.from_arrow(
        pa.array(["C", "e", "I", "j", "m", "q", None, "T", None, "w"])
    )

    expected = pa.array(
        [
            elem.endswith(target) if elem is not None else None
            for elem, target in zip(
                plc.interop.to_arrow(plc_col).to_pylist(),
                plc.interop.to_arrow(target_col).to_pylist(),
            )
        ],
        type=pa.bool_(),
    )

    got = plc.strings.find.ends_with(plc_col, target_col)
    assert_column_eq(got, expected)
