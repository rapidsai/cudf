# Copyright (c) 2024, NVIDIA CORPORATION.

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc
from cudf._lib.scalar import DeviceScalar


@pytest.fixture
def pa_data_col():
    return pa.array(
        [
            "abc123",
            "ABC123",
            "aBc123",
            "",
            " ",
            None,
            "a",
            None,
            "abc123",
            "ABC123",
            "aBc123",
            "",
            " ",
            None,
            "a",
            None,
            "abc123",
            "ABC123",
            "aBc123",
            "",
            " ",
            None,
            "a",
            None,
            "abc123",
            "ABC123",
            "aBc123",
            "",
            " ",
            None,
            "a",
            None,
            "abc123",
            "ABC123",
            "aBc123",
            "",
            " ",
            None,
            "a",
            None,
        ]
    )


@pytest.fixture
def plc_data_col(pa_data_col):
    return plc.interop.from_arrow(pa_data_col)


@pytest.fixture
def pa_target_col():
    return pa.array(
        [
            "a",
            "B",
            "x",
            "1",
            " ",
            "a",
            None,
            None,  # find
            "a",
            "B",
            "x",
            "1",
            " ",
            "a",
            None,
            None,  # rfind
            "ab",
            "12",
            "BC",
            "",
            " ",
            "a",
            None,
            None,  # contains
            "ab",
            "ABC",
            "AB",
            "",
            " ",
            "a",
            None,
            None,  # starts_with
            "3",
            "23",
            "a23",
            "",
            " ",
            "a",
            None,
            None,  # ends_with
        ]
    )


@pytest.fixture
def plc_target_col(pa_target_col):
    return plc.interop.from_arrow(pa_target_col)


@pytest.fixture(params=["a", " ", "A", "Ab", "23"])
def plc_target_scalar(request):
    return plc.interop.from_arrow(pa.scalar(request.param, type=pa.string()))


def test_find(pa_data_col, plc_data_col, plc_target_scalar):
    got = plc.strings.find.find(plc_data_col, plc_target_scalar, 0, -1)

    expected = pa.array(
        [
            elem.find(plc.interop.to_arrow(plc_target_scalar).as_py())
            if elem is not None
            else None
            for elem in pa_data_col.to_pylist()
        ],
        type=pa.int32(),
    )

    assert_column_eq(got, expected)


def test_find_column(pa_data_col, pa_target_col, plc_data_col, plc_target_col):
    expected = pa.array(
        [
            elem.find(target) if not (elem is None or target is None) else None
            for elem, target in zip(
                pa_data_col.to_pylist(),
                pa_target_col.to_pylist(),
            )
        ],
        type=pa.int32(),
    )

    got = plc.strings.find.find(plc_data_col, plc_target_col, 0)
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


def test_contains_column(plc_col, contains_target_column):
    expected = pa.array(
        [
            target in elem if elem is not None else None
            for elem, target in zip(
                plc.interop.to_arrow(plc_col).to_pylist(),
                plc.interop.to_arrow(contains_target_column).to_pylist(),
            )
        ],
        type=pa.bool_(),
    )

    got = plc.strings.find.contains(plc_col, contains_target_column)
    assert_column_eq(got, expected)


@pytest.mark.parametrize("target", ["A", "", "Ab"])
def test_starts_with(plc_col, target):
    got = plc.strings.find.starts_with(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value
    )
    expected = pa.compute.starts_with(plc.interop.to_arrow(plc_col), target)
    assert_column_eq(got, expected)


def test_starts_with_column(plc_col, starts_with_target_column):
    expected = pa.array(
        [
            elem.startswith(target) if elem is not None else None
            for elem, target in zip(
                plc.interop.to_arrow(plc_col).to_pylist(),
                plc.interop.to_arrow(starts_with_target_column).to_pylist(),
            )
        ],
        type=pa.bool_(),
    )

    got = plc.strings.find.starts_with(plc_col, starts_with_target_column)
    assert_column_eq(got, expected)


@pytest.mark.parametrize("target", ["C", "bC", "BC", ""])
def test_ends_with(plc_col, target):
    got = plc.strings.find.ends_with(
        plc_col, DeviceScalar(target, dtype=np.dtype("object")).c_value
    )
    expected = pa.compute.ends_with(plc.interop.to_arrow(plc_col), target)
    assert_column_eq(got, expected)


def test_ends_with_column(plc_col, ends_with_target_column):
    expected = pa.array(
        [
            elem.endswith(target) if elem is not None else None
            for elem, target in zip(
                plc.interop.to_arrow(plc_col).to_pylist(),
                plc.interop.to_arrow(ends_with_target_column).to_pylist(),
            )
        ],
        type=pa.bool_(),
    )

    got = plc.strings.find.ends_with(plc_col, ends_with_target_column)
    assert_column_eq(got, expected)
