# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


@pytest.fixture(scope="module")
def is_title_data():
    data = [
        "leopard",
        "Golden Eagle",
        "SNAKE",
        "",
        "!A",
        "hello World",
        "A B C",
        "#",
        "AƻB",
        "Ⓑⓖ",
        "Art of War",
    ]
    return pa.array(data)


@pytest.fixture(scope="module")
def title_data():
    data = [
        None,
        "The quick bRoWn fox juMps over the laze DOG",
        '123nr98nv9rev!$#INF4390v03n1243<>?}{:-"',
        "accénted",
    ]
    return pa.array(data)


def test_capitalize(title_data):
    plc_col = plc.interop.from_arrow(title_data)
    got = plc.strings.capitalize.capitalize(plc_col)
    expected = pa.compute.utf8_capitalize(title_data)
    assert_column_eq(got, expected)


def test_title(title_data):
    plc_col = plc.interop.from_arrow(title_data)
    got = plc.strings.capitalize.title(plc_col)
    expected = pa.compute.utf8_title(title_data)
    assert_column_eq(got, expected)


def test_is_title(is_title_data):
    plc_col = plc.interop.from_arrow(is_title_data)
    got = plc.strings.capitalize.is_title(plc_col)
    expected = pa.compute.utf8_is_title(is_title_data)
    assert_column_eq(got, expected)
