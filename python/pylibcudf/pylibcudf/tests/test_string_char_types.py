# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pylibcudf as plc


def test_all_characters_of_type():
    pa_array = pa.array(["1", "A"])
    plc_result = plc.strings.char_types.all_characters_of_type(
        plc.interop.from_arrow(pa_array),
        plc.strings.char_types.StringCharacterTypes.ALPHA,
        plc.strings.char_types.StringCharacterTypes.ALL_TYPES,
    )
    pa_result = plc.interop.to_arrow(plc_result)
    pa_expected = pa.chunked_array([pc.utf8_is_alpha(pa_array)])
    assert pa_result.equals(pa_expected)


def test_filter_characters_of_type():
    pa_array = pa.array(["=A="])
    plc_result = plc.strings.char_types.filter_characters_of_type(
        plc.interop.from_arrow(pa_array),
        plc.strings.char_types.StringCharacterTypes.ALPHANUM,
        plc.interop.from_arrow(pa.scalar(" ")),
        plc.strings.char_types.StringCharacterTypes.ALL_TYPES,
    )
    pa_result = plc.interop.to_arrow(plc_result)
    pa_expected = pa.chunked_array([pc.replace_substring(pa_array, "A", " ")])
    assert pa_result.equals(pa_expected)
