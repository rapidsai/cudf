# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def str_data():
    pa_data = pa.array(
        [
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
    )
    return pa_data, plc.interop.from_arrow(pa_data)


def test_capitalize(str_data):
    pa_data, plc_data = str_data
    got = plc.strings.capitalize.capitalize(plc_data)
    expected = pc.utf8_capitalize(pa_data)
    assert_column_eq(expected, got)


def test_title(str_data):
    pa_data, plc_data = str_data
    got = plc.strings.capitalize.title(
        plc_data, plc.strings.char_types.StringCharacterTypes.CASE_TYPES
    )
    expected = pc.utf8_title(pa_data)
    assert_column_eq(expected, got)


def test_is_title(str_data):
    pa_data, plc_data = str_data
    got = plc.strings.capitalize.is_title(plc_data)
    expected = pc.utf8_is_title(pa_data)
    assert_column_eq(expected, got)
