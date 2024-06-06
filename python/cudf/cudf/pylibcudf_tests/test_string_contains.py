# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


@pytest.fixture(scope="module")
def pa_target_col():
    return pa.array(
        ["AbC", "de", "FGHI", "j", "kLm", "nOPq", None, "RsT", None, "uVw"]
    )


@pytest.fixture(scope="module")
def plc_target_col(pa_target_col):
    return plc.interop.from_arrow(pa_target_col)


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


def test_contains_re(
    pa_target_col, plc_target_col, pa_target_scalar, plc_target_pat
):
    got = plc.strings.contains.contains_re(plc_target_col, plc_target_pat)
    expected = pa.compute.match_substring_regex(
        pa_target_col, pa_target_scalar.as_py()
    )
    assert_column_eq(got, expected)
