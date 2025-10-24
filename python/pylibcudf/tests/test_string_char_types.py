# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.compute as pc
from utils import assert_column_eq

import pylibcudf as plc


def test_all_characters_of_type():
    pa_array = pa.array(["1", "A"])
    got = plc.strings.char_types.all_characters_of_type(
        plc.Column.from_arrow(pa_array),
        plc.strings.char_types.StringCharacterTypes.ALPHA,
        plc.strings.char_types.StringCharacterTypes.ALL_TYPES,
    )
    expect = pc.utf8_is_alpha(pa_array)
    assert_column_eq(expect, got)


def test_filter_characters_of_type():
    pa_array = pa.array(["=A="])
    got = plc.strings.char_types.filter_characters_of_type(
        plc.Column.from_arrow(pa_array),
        plc.strings.char_types.StringCharacterTypes.ALPHANUM,
        plc.Scalar.from_arrow(pa.scalar(" ")),
        plc.strings.char_types.StringCharacterTypes.ALL_TYPES,
    )
    expect = pc.replace_substring(pa_array, "A", " ")
    assert_column_eq(expect, got)
