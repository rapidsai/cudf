# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import string

import pandas as pd
import pytest

import nvstrings
from utils import assert_eq


@pytest.mark.parametrize(
    "table",
    [
        [],
        pytest.param(
            [["e", "a"]],
            marks=[
                pytest.mark.xfail(
                    reason="""
                  Pandas series requires ordinal mapping
                                          """
                )
            ],
        ),
        pytest.param(
            [["e", "é"]],
            marks=[
                pytest.mark.xfail(
                    reason="""
                  Pandas series requires ordinal mapping
                                          """
                )
            ],
        ),
        pytest.param(
            [["é", "e"]],
            marks=[
                pytest.mark.xfail(
                    reason="""
                  Pandas series requires ordinal mapping
                  """
                )
            ],
        ),
        pytest.param(
            [["o", None]],
            marks=[
                pytest.mark.xfail(
                    reason="""
                  Pandas series requires ordinal mapping
                                          """
                )
            ],
        ),
    ],
)
def test_translate_from_list(table):
    s = ["hello", "there", "world", "accéntéd", None, ""]
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    got = strs.translate(table)
    expected = pstrs.str.translate(table)
    assert_eq(got.to_host(), expected)


@pytest.mark.parametrize(
    "table",
    [
        {},
        str.maketrans("e", "a"),
        str.maketrans("elh", "ELH"),
        str.maketrans("", "", string.punctuation),
        str.maketrans(string.punctuation, " " * len(string.punctuation)),
    ],
)
def test_translate_from_ordinal(table):
    s = ["hello", "there", "world", "accéntéd", None, ""]
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    got = strs.translate(table)
    expected = pstrs.str.translate(table)
    assert_eq(got.to_host(), expected)

    s = [
        "This, of course, is only an example!",  # noqa E501
        "And; will have @all the #punctuation that $money can buy.",  # noqa E501
        "The %percent & the *star along with the (parenthesis) with dashes-and-under_lines.",  # noqa E501
        "Equations: 3+3=6; 3/4 < 1 and > 0",
    ]  # noqa E501
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    got = strs.translate(table)
    expected = pstrs.str.translate(table)
    assert_eq(got.to_host(), expected)
