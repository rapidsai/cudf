# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import numpy as np

import nvstrings
from utils import assert_eq


def test_hash():
    s = nvstrings.to_device(
        [
            "1234",
            "5678",
            "90",
            None,
            "-876",
            "543.2",
            "-0.12",
            ".55",
            "-.002",
            "",
            "de",
            "abc123",
            "123abc",
            "456e",
            "-1.78e+5",
        ]
    )
    got = s.hash()
    expected = [
        1762063109,
        3008518326,
        3419725934,
        None,
        1225421472,
        2952354928,
        2093756495,
        1292375090,
        2098378342,
        1257683291,
        3758453927,
        213530502,
        2957649541,
        4248160425,
        2735531987,
    ]
    assert_eq(got, expected)


def test_isalnum():
    s = nvstrings.to_device(
        [
            "1234567890",
            "de",
            "1.75",
            "-34",
            "+9.8",
            "7¼",
            "x³",
            "2³",
            "12⅝",
            "",
            "\t\r\n ",
        ]
    )
    got = s.isalnum()
    expected = [
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert_eq(got, expected)


def test_isalpha():
    s = nvstrings.to_device(
        [
            "1234567890",
            "de",
            "1.75",
            "-34",
            "+9.8",
            "7¼",
            "x³",
            "2³",
            "12⅝",
            "",
            "\t\r\n ",
        ]
    )
    got = s.isalpha()
    expected = [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert_eq(got, expected)


def test_isdigit():
    s = nvstrings.to_device(
        [
            "1234567890",
            "de",
            "1.75",
            "-34",
            "+9.8",
            "7¼",
            "x³",
            "2³",
            "12⅝",
            "",
            "\t\r\n ",
        ]
    )
    got = s.isdigit()
    expected = [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert_eq(got, expected)


def test_isdecimal():
    s = nvstrings.to_device(
        [
            "1234567890",
            "de",
            "1.75",
            "-34",
            "+9.8",
            "7¼",
            "x³",
            "2³",
            "12⅝",
            "",
            "\t\r\n ",
        ]
    )
    got = s.isdecimal()
    expected = [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert_eq(got, expected)


def test_isspace():
    s = nvstrings.to_device(
        [
            "1234567890",
            "de",
            "1.75",
            "-34",
            "+9.8",
            "7¼",
            "x³",
            "2³",
            "12⅝",
            "",
            "\t\r\n ",
        ]
    )
    got = s.isspace()
    expected = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert_eq(got, expected)


def test_isnumeric():
    s = nvstrings.to_device(
        [
            "1234567890",
            "de",
            "1.75",
            "-34",
            "+9.8",
            "7¼",
            "x³",
            "2³",
            "12⅝",
            "",
            "\t\r\n ",
        ]
    )
    got = s.isnumeric()
    expected = [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    assert_eq(got, expected)


def test_stoi():
    s = nvstrings.to_device(
        [
            "1234",
            "5678",
            "90",
            None,
            "-876",
            "543.2",
            "-0.12",
            ".55",
            "-.002",
            "",
            "de",
            "abc123",
            "123abc",
            "456e",
            "-1.78e+5",
        ]
    )
    got = s.stoi()
    expected = [
        1234,
        5678,
        90,
        None,
        -876,
        543,
        0,
        0,
        0,
        0,
        0,
        0,
        123,
        456,
        -1,
    ]
    assert_eq(got, expected)


def test_stol():
    s = nvstrings.to_device(
        [
            "1234",
            "5678",
            "90",
            None,
            "-876",
            "543.2",
            "-0.12",
            "2.55",
            "-.002",
            "",
            "de",
            "abc123",
            "123abc",
            "456e",
            "-1.78e+5",
        ]
    )
    got = s.stol()
    expected = [
        1234,
        5678,
        90,
        None,
        -876,
        543,
        0,
        2,
        0,
        0,
        0,
        0,
        123,
        456,
        -1,
    ]
    assert_eq(got, expected)


def test_stof():
    s = nvstrings.to_device(
        [
            "1234",
            "5678",
            "90",
            None,
            "-876",
            "543.2",
            "-0.12",
            ".55",
            "-.002",
            "",
            "de",
            "abc123",
            "123abc",
            "456e",
            "-1.78e+5",
        ]
    )
    got = s.stof()
    expected = [
        1234.0,
        5678.0,
        90.0,
        None,
        -876.0,
        543.2000122070312,
        -0.11999999731779099,
        0.550000011920929,
        -0.0020000000949949026,
        0.0,
        0.0,
        0.0,
        123.0,
        456.0,
        -178000.0,
    ]
    assert_eq(got, expected)


def test_stod():
    s = nvstrings.to_device(
        [
            "1234",
            "5678",
            "90",
            None,
            "-876",
            "543.2",
            "-0.12",
            "2.553",
            "-.002",
            "",
            "de",
            "abc123",
            "123abc",
            "456e",
            "-1.78e+5",
            "-122.33644782",
        ]
    )
    got = s.stod()
    expected = [
        1234.0,
        5678.0,
        90.0,
        None,
        -876.0,
        543.2,
        -0.12,
        2.553,
        -0.002,
        0.0,
        0.0,
        0.0,
        123.0,
        456.0,
        -178000.0,
        -122.33644781999999,
    ]
    assert_eq(got, expected)


def test_htoi():
    s = nvstrings.to_device(["1234", "ABCDEF", "1A2", "cafe"])
    got = s.htoi()
    expected = [4660, 11259375, 418, 51966]
    assert_eq(got, expected)


def test_itos():
    s = [0, 103, 1053, 8395739]
    got = nvstrings.itos(s)
    expected = nvstrings.to_device(["0", "103", "1053", "8395739"])
    assert_eq(got, expected)


def test_ltos():
    s = [0, 103, -2548485929, 8395794248339]
    got = nvstrings.ltos(s)
    expected = nvstrings.to_device(
        ["0", "103", "-2548485929", "8395794248339"]
    )
    assert_eq(got, expected)


def test_ftos():
    s = np.array(
        [0, 103, -254848.5929, 8395794.248339, np.nan, np.inf],
        dtype=np.float32,
    )
    got = nvstrings.ftos(s)
    expected = nvstrings.to_device(
        ["0.0", "103.0", "-254848.5938", "8395794.0", "NaN", "Inf"]
    )
    assert_eq(got, expected)


def test_dtos():
    s = np.array(
        [0, 103342.313, -25.4294, 839542223232.794248339, np.nan],
        dtype=np.float64,
    )
    got = nvstrings.dtos(s)
    expected = nvstrings.to_device(
        ["0.0", "103342.313", "-25.4294", "8.395422232e+11", "NaN"]
    )
    assert_eq(got, expected)


def test_ip2int():
    s = nvstrings.to_device(
        [
            "192.168.0.1",
            "10.0.0.1",
            None,
            "",
            "hello",
            "41.186.0.1",
            "41.197.0.1",
        ]
    )
    got = s.ip2int()
    expected = [3232235521, 167772161, None, 0, 0, 700055553, 700776449]
    assert_eq(got, expected)


def test_int2ip():
    ints = [3232235521, 167772161, None, 0, 0, 700055553, 700776449]
    got = nvstrings.int2ip(ints)
    expected = [
        "192.168.0.1",
        "10.0.0.1",
        "0.0.0.0",
        "0.0.0.0",
        "0.0.0.0",
        "41.186.0.1",
        "41.197.0.1",
    ]
    assert_eq(got, expected)


def test_to_booleans():
    s = nvstrings.to_device(["true", "false", None, "", "true", "True"])

    got = s.to_booleans()
    expected = [False, False, None, False, False, True]
    assert_eq(got, expected)

    got = s.to_booleans(true="true")
    expected = [True, False, None, False, True, False]
    assert_eq(got, expected)


def test_from_booleans():
    s = [True, False, False, True]
    got = nvstrings.from_booleans(s)
    expected = ["True", "False", "False", "True"]
    assert_eq(got, expected)

    got = nvstrings.from_booleans(s, nulls=[11])
    expected = ["True", "False", None, "True"]
    assert_eq(got, expected)


def test_is_empty():
    s = nvstrings.to_device(["true", "false", None, "", "true", "True"])
    got = s.is_empty()
    expected = [False, False, None, True, False, False]
    assert_eq(got, expected)


def test_copy():
    s = nvstrings.to_device(["true", "false", None, "", "true", "True"])
    s1 = s.copy()
    assert_eq(s, s1)


def test_to_host():
    s = nvstrings.to_device(["true", "false", None, "", "true", "True"])
    got = s.to_host()
    expected = ["true", "false", None, "", "true", "True"]
    assert_eq(got, expected)


def test_to_device():
    pass
