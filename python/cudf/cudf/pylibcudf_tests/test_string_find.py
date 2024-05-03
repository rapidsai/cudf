# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


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


def test_rfind(pa_data_col, plc_data_col, plc_target_scalar):
    py_target = plc.interop.to_arrow(plc_target_scalar).as_py()

    got = plc.strings.find.rfind(plc_data_col, plc_target_scalar, 0, -1)

    expected = pa.array(
        [
            elem.rfind(py_target)
            if not (elem is None or py_target is None)
            else None
            for elem in pa_data_col.to_pylist()
        ],
        type=pa.int32(),
    )

    assert_column_eq(got, expected)


def test_contains(pa_data_col, plc_data_col, plc_target_scalar):
    py_target = plc.interop.to_arrow(plc_target_scalar).as_py()

    got = plc.strings.find.contains(plc_data_col, plc_target_scalar)
    expected = pa.array(
        [
            py_target in elem
            if not (elem is None or py_target is None)
            else None
            for elem in pa_data_col.to_pylist()
        ],
        type=pa.bool_(),
    )

    assert_column_eq(got, expected)


def test_contains_column(
    pa_data_col, pa_target_col, plc_data_col, plc_target_col
):
    def libcudf_logic(st, target):
        if st is None:
            return None
        elif target is None:
            return False
        else:
            return target in st

    expected = pa.array(
        [
            libcudf_logic(elem, target)
            for elem, target in zip(
                pa_data_col.to_pylist(),
                pa_target_col.to_pylist(),
            )
        ],
        type=pa.bool_(),
    )

    got = plc.strings.find.contains(plc_data_col, plc_target_col)
    assert_column_eq(got, expected)


def test_starts_with(pa_data_col, plc_data_col, plc_target_scalar):
    py_target = plc.interop.to_arrow(plc_target_scalar).as_py()
    got = plc.strings.find.starts_with(plc_data_col, plc_target_scalar)
    expected = pa.compute.starts_with(pa_data_col, py_target)
    assert_column_eq(got, expected)


def test_starts_with_column(
    pa_data_col, pa_target_col, plc_data_col, plc_target_col
):
    def libcudf_logic(st, target):
        if st is None:
            return None
        elif target is None:
            return False
        else:
            return st.startswith(target)

    expected = pa.array(
        [
            libcudf_logic(elem, target)
            for elem, target in zip(
                pa_data_col.to_pylist(),
                pa_target_col.to_pylist(),
            )
        ],
        type=pa.bool_(),
    )

    got = plc.strings.find.starts_with(plc_data_col, plc_target_col)
    assert_column_eq(got, expected)


def test_ends_with(pa_data_col, plc_data_col, plc_target_scalar):
    py_target = plc.interop.to_arrow(plc_target_scalar).as_py()
    got = plc.strings.find.ends_with(plc_data_col, plc_target_scalar)
    expected = pa.compute.ends_with(pa_data_col, py_target)
    assert_column_eq(got, expected)


def test_ends_with_column(
    pa_data_col, pa_target_col, plc_data_col, plc_target_col
):
    def libcudf_logic(st, target):
        if st is None:
            return None
        elif target is None:
            return False
        else:
            return st.endswith(target)

    expected = pa.array(
        [
            libcudf_logic(elem, target)
            for elem, target in zip(
                pa_data_col.to_pylist(),
                pa_target_col.to_pylist(),
            )
        ],
        type=pa.bool_(),
    )

    got = plc.strings.find.ends_with(plc_data_col, plc_target_col)
    assert_column_eq(got, expected)
