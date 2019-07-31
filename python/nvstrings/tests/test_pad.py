# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings
from utils import assert_eq


def test_rjust():
    strs = nvstrings.to_device(["abc", "Def", None, "jLl"])
    got = strs.rjust(4)
    expected = [" abc", " Def", None, " jLl"]
    assert_eq(got, expected)


def test_pad():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "1234", "-1234", None, "accént", ""]
    )
    got = strs.pad(5)
    expected = [
        "hello",
        "there",
        "world",
        " 1234",
        "-1234",
        None,
        "accént",
        "     ",
    ]
    assert_eq(got, expected)

    # pad right only
    got = strs.pad(7, "right")
    expected = [
        "hello  ",
        "there  ",
        "world  ",
        "1234   ",
        "-1234  ",
        None,
        "accént ",
        "       ",
    ]
    assert_eq(got, expected)

    # pad both with a specific character
    strs = nvstrings.to_device(
        ["hello", "there", "world", "1234", "-1234", None, "accént", ""]
    )
    got = strs.pad(9, "both", ".")
    expected = [
        "..hello..",
        "..there..",
        "..world..",
        "..1234...",
        "..-1234..",
        None,
        ".accént..",
        ".........",
    ]
    assert_eq(got, expected)


def test_ljust():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "1234", "-1234", None, "accént", ""]
    )
    got = strs.ljust(7)
    expected = [
        "hello  ",
        "there  ",
        "world  ",
        "1234   ",
        "-1234  ",
        None,
        "accént ",
        "       ",
    ]
    assert_eq(got, expected)


def test_center():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "1234", "-1234", None, "accént", ""]
    )
    got = strs.center(10, " ")
    expected = [
        "  hello   ",
        "  there   ",
        "  world   ",
        "   1234   ",
        "  -1234   ",
        None,
        "  accént  ",
        "          ",
    ]
    assert_eq(got, expected)


def test_zfill():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "1234", "-1234", None, "accént", ""]
    )
    got = strs.zfill(6)
    expected = [
        "0hello",
        "0there",
        "0world",
        "001234",
        "-01234",
        None,
        "accént",
        "000000",
    ]
    assert_eq(got, expected)


def test_repeat():
    strs = nvstrings.to_device(
        ["hello", "there", "world", "1234", "-1234", None, "accént", ""]
    )
    got = strs.repeat(6)
    expected = [
        "hellohellohellohellohellohello",
        "theretheretheretheretherethere",
        "worldworldworldworldworldworld",
        "123412341234123412341234",
        "-1234-1234-1234-1234-1234-1234",
        None,
        "accéntaccéntaccéntaccéntaccéntaccént",
        "",
    ]
    assert_eq(got, expected)
