# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pylibcudf as plc
import pytest
from utils import assert_column_eq


@pytest.fixture(scope="module")
def target_col():
    pa_array = pa.array(
        ["AbC", "de", "FGHI", "j", "kLm", "nOPq", None, "RsT", None, "uVw"]
    )
    return pa_array, plc.interop.from_arrow(pa_array)


@pytest.fixture(
    params=[
        "A",
        "de",
        ".*",
        "^a",
        "^A",
        "[^a-z]",
        "[a-z]{3,}",
        "^[A-Z]{2,}",
        "j|u",
    ],
    scope="module",
)
def pa_target_scalar(request):
    return pa.scalar(request.param, type=pa.string())


@pytest.fixture(scope="module")
def plc_target_pat(pa_target_scalar):
    prog = plc.strings.regex_program.RegexProgram.create(
        pa_target_scalar.as_py(), plc.strings.regex_flags.RegexFlags.DEFAULT
    )
    return prog


def test_contains_re(target_col, pa_target_scalar, plc_target_pat):
    pa_target_col, plc_target_col = target_col
    got = plc.strings.contains.contains_re(plc_target_col, plc_target_pat)
    expected = pc.match_substring_regex(
        pa_target_col, pa_target_scalar.as_py()
    )
    assert_column_eq(got, expected)
