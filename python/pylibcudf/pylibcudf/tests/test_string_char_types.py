# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
from utils import assert_column_eq

import pylibcudf as plc


def test_all_characters_of_type():
    pa_array = pa.array(["1", "A"])
    result = plc.strings.char_types.all_characters_of_type(
        plc.interop.from_arrow(pa_array),
        plc.strings.char_types.StringCharacterTypes.ALPHA,
        plc.strings.char_types.StringCharacterTypes.ALL_TYPES,
    )
    expected = pc.utf8_is_alpha(pa_array)
    assert_column_eq(result, expected)


def test_filter_characters_of_type():
    pa_array = pa.array(["=A="])
    result = plc.strings.char_types.filter_characters_of_type(
        plc.interop.from_arrow(pa_array),
        plc.strings.char_types.StringCharacterTypes.ALPHANUM,
        plc.interop.from_arrow(pa.scalar(" ")),
        plc.strings.char_types.StringCharacterTypes.ALL_TYPES,
    )
    expected = pc.replace_substring(pa_array, "A", " ")
    assert_column_eq(result, expected)
