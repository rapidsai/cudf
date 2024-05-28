# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


@pytest.fixture(scope="module")
def pa_data():
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
        "The quick bRoWn fox juMps over the laze DOG",
        '123nr98nv9rev!$#INF4390v03n1243<>?}{:-"',
        "accénted",
        None,
    ]
    return pa.array(data)


@pytest.fixture(scope="module")
def plc_data(pa_data):
    return plc.interop.from_arrow(pa_data)


def test_capitalize(plc_data, pa_data):
    got = plc.strings.capitalize.capitalize(plc_data)
    expected = pa.compute.utf8_capitalize(pa_data)
    assert_column_eq(got, expected)


def test_title(plc_data, pa_data):
    got = plc.strings.capitalize.title(
        plc_data, plc.strings.char_types.StringCharacterTypes.CASE_TYPES
    )
    expected = pa.compute.utf8_title(pa_data)
    assert_column_eq(got, expected)


def test_is_title(plc_data, pa_data):
    got = plc.strings.capitalize.is_title(plc_data)
    expected = pa.compute.utf8_is_title(pa_data)
    assert_column_eq(got, expected)
