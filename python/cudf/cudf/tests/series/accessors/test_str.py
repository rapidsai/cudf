# Copyright (c) 2025, NVIDIA CORPORATION.

import json
import re
import urllib.parse
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf import concat
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
    expect_warning_if,
)


def raise_builder(flags, exceptions):
    if any(flags):
        return pytest.raises(exceptions)
    else:
        return does_not_raise()


@pytest.fixture(
    params=[
        ["AbC", "de", "FGHI", "j", "kLm"],
        ["nOPq", None, "RsT", None, "uVw"],
        [None, None, None, None, None],
    ],
    ids=["no_nulls", "some_nulls", "all_nulls"],
)
def data(request):
    return request.param


@pytest.fixture(
    params=[None, [10, 11, 12, 13, 14]], ids=["None_index", "Set_index"]
)
def index(request):
    return request.param


@pytest.fixture
def ps_gs(data, index):
    ps = pd.Series(data, index=index, dtype="str", name="nice name")
    gs = cudf.Series(data, index=index, dtype="str", name="nice name")
    return (ps, gs)


def test_getitem_out_of_bounds():
    data = ["123", "12", "1"]
    pd_ser = pd.Series(data)
    cudf_ser = cudf.Series(data)
    expected = pd_ser.str[2]
    result = cudf_ser.str[2]
    assert_eq(result, expected)

    expected = pd_ser.str[-2]
    result = cudf_ser.str[-2]
    assert_eq(result, expected)


@pytest.mark.parametrize("method", ["startswith", "endswith"])
@pytest.mark.parametrize("pat", [None, (1, 2), pd.Series([1])])
def test_startsendwith_invalid_pat(method, pat):
    ser = cudf.Series(["1"])
    with pytest.raises(TypeError):
        getattr(ser.str, method)(pat)


@pytest.mark.parametrize("method", ["rindex", "index"])
def test_index_int64_pandas_compat(method):
    data = ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"]
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(cudf.Series(data).str, method)("E", 4, 8)
    expected = getattr(pd.Series(data).str, method)("E", 4, 8)
    assert_eq(result, expected)


def test_replace_invalid_scalar_repl():
    ser = cudf.Series(["1"])
    with pytest.raises(TypeError):
        ser.str.replace("1", 2)


def test_string_methods_setattr():
    ser = cudf.Series(["ab", "cd", "ef"])
    pser = ser.to_pandas()

    assert_exceptions_equal(
        lfunc=ser.str.__setattr__,
        rfunc=pser.str.__setattr__,
        lfunc_args_and_kwargs=(("a", "b"),),
        rfunc_args_and_kwargs=(("a", "b"),),
    )


@pytest.mark.parametrize(
    "data",
    [
        [
            """
            {
                "store":{
                    "book":[
                        {
                            "category":"reference",
                            "author":"Nigel Rees",
                            "title":"Sayings of the Century",
                            "price":8.95
                        },
                        {
                            "category":"fiction",
                            "author":"Evelyn Waugh",
                            "title":"Sword of Honour",
                            "price":12.99
                        }
                    ]
                }
            }
            """
        ],
        [
            """
            {
                "store":{
                    "book":[
                        {
                            "category":"reference",
                            "author":"Nigel Rees",
                            "title":"Sayings of the Century",
                            "price":8.95
                        }
                    ]
                }
            }
            """,
            """
            {
                "store":{
                    "book":[
                        {
                            "category":"fiction",
                            "author":"Evelyn Waugh",
                            "title":"Sword of Honour",
                            "price":12.99
                        }
                    ]
                }
            }
            """,
        ],
    ],
)
def test_string_get_json_object_n(data):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(
        json.loads(gs.str.get_json_object("$.store")[0]),
        ps.apply(lambda x: json.loads(x)["store"])[0],
    )
    assert_eq(
        json.loads(gs.str.get_json_object("$.store.book")[0]),
        ps.apply(lambda x: json.loads(x)["store"]["book"])[0],
    )
    assert_eq(
        gs.str.get_json_object("$.store.book[0].category"),
        ps.apply(lambda x: json.loads(x)["store"]["book"][0]["category"]),
    )


@pytest.mark.parametrize(
    "json_path", ["$.store", "$.store.book", "$.store.book[*].category", " "]
)
def test_string_get_json_object_empty_json_strings(json_path):
    gs = cudf.Series(
        [
            """
            {
                "":{
                    "":[
                        {
                            "":"",
                            "":"",
                            "":""
                        },
                        {
                            "":"fiction",
                            "":"",
                            "title":""
                        }
                    ]
                }
            }
            """
        ]
    )

    got = gs.str.get_json_object(json_path)
    expect = cudf.Series([None], dtype="object")

    assert_eq(got, expect)


@pytest.mark.parametrize("json_path", ["a", ".", "/.store"])
def test_string_get_json_object_invalid_JSONPath(json_path):
    gs = cudf.Series(
        [
            """
            {
                "store":{
                    "book":[
                        {
                            "category":"reference",
                            "author":"Nigel Rees",
                            "title":"Sayings of the Century",
                            "price":8.95
                        },
                        {
                            "category":"fiction",
                            "author":"Evelyn Waugh",
                            "title":"Sword of Honour",
                            "price":12.99
                        }
                    ]
                }
            }
            """
        ]
    )

    with pytest.raises(ValueError):
        gs.str.get_json_object(json_path)


def test_string_get_json_object_allow_single_quotes():
    gs = cudf.Series(
        [
            """
            {
                "store":{
                    "book":[
                        {
                            'author':"Nigel Rees",
                            "title":'Sayings of the Century',
                            "price":8.95
                        },
                        {
                            "category":"fiction",
                            "author":"Evelyn Waugh",
                            'title':"Sword of Honour",
                            "price":12.99
                        }
                    ]
                }
            }
            """
        ]
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[0].author", allow_single_quotes=True
        ),
        cudf.Series(["Nigel Rees"]),
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[*].title", allow_single_quotes=True
        ),
        cudf.Series(["['Sayings of the Century',\"Sword of Honour\"]"]),
    )

    assert_eq(
        gs.str.get_json_object(
            "$.store.book[0].author", allow_single_quotes=False
        ),
        cudf.Series([None]),
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[*].title", allow_single_quotes=False
        ),
        cudf.Series([None]),
    )


def test_string_get_json_object_strip_quotes_from_single_strings():
    gs = cudf.Series(
        [
            """
            {
                "store":{
                    "book":[
                        {
                            "author":"Nigel Rees",
                            "title":"Sayings of the Century",
                            "price":8.95
                        },
                        {
                            "category":"fiction",
                            "author":"Evelyn Waugh",
                            "title":"Sword of Honour",
                            "price":12.99
                        }
                    ]
                }
            }
            """
        ]
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[0].author", strip_quotes_from_single_strings=True
        ),
        cudf.Series(["Nigel Rees"]),
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[*].title", strip_quotes_from_single_strings=True
        ),
        cudf.Series(['["Sayings of the Century","Sword of Honour"]']),
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[0].author", strip_quotes_from_single_strings=False
        ),
        cudf.Series(['"Nigel Rees"']),
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[*].title", strip_quotes_from_single_strings=False
        ),
        cudf.Series(['["Sayings of the Century","Sword of Honour"]']),
    )


def test_string_get_json_object_missing_fields_as_nulls():
    gs = cudf.Series(
        [
            """
            {
                "store":{
                    "book":[
                        {
                            "author":"Nigel Rees",
                            "title":"Sayings of the Century",
                            "price":8.95
                        },
                        {
                            "category":"fiction",
                            "author":"Evelyn Waugh",
                            "title":"Sword of Honour",
                            "price":12.99
                        }
                    ]
                }
            }
            """
        ]
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[0].category", missing_fields_as_nulls=True
        ),
        cudf.Series(["null"]),
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[*].category", missing_fields_as_nulls=True
        ),
        cudf.Series(['[null,"fiction"]']),
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[0].category", missing_fields_as_nulls=False
        ),
        cudf.Series([None]),
    )
    assert_eq(
        gs.str.get_json_object(
            "$.store.book[*].category", missing_fields_as_nulls=False
        ),
        cudf.Series(['["fiction"]']),
    )


def test_str_join_lists_error():
    sr = cudf.Series([["a", "a"], ["b"], ["c"]])

    with pytest.raises(
        ValueError, match="sep_na_rep cannot be defined when `sep` is scalar."
    ):
        sr.str.join(sep="-", sep_na_rep="-")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "string_na_rep should be a string scalar, got [10, 20] of type "
            ": <class 'list'>"
        ),
    ):
        sr.str.join(string_na_rep=[10, 20])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "sep should be of similar size to the series, got: 2, expected: 3"
        ),
    ):
        sr.str.join(sep=["=", "-"])

    with pytest.raises(
        TypeError,
        match=re.escape(
            "sep_na_rep should be a string scalar, got "
            "['na'] of type: <class 'list'>"
        ),
    ):
        sr.str.join(sep=["-", "+", "."], sep_na_rep=["na"])

    with pytest.raises(
        TypeError,
        match=re.escape(
            "sep should be an str, array-like or Series object, "
            "found <class 'cudf.core.dataframe.DataFrame'>"
        ),
    ):
        sr.str.join(sep=cudf.DataFrame())


@pytest.mark.parametrize(
    "sr,sep,string_na_rep,sep_na_rep,expected",
    [
        (
            [["a", "a"], ["b"], ["c"]],
            "-",
            None,
            None,
            ["a-a", "b", "c"],
        ),
        (
            [["a", "b"], [None], [None, "hello", None, "world"]],
            "__",
            "=",
            None,
            ["a__b", None, "=__hello__=__world"],
        ),
        (
            [
                ["a", None, "b"],
                [None],
                [None, "hello", None, "world"],
                None,
            ],
            ["-", "_", "**", "!"],
            None,
            None,
            ["a--b", None, "**hello****world", None],
        ),
        (
            [
                ["a", None, "b"],
                [None],
                [None, "hello", None, "world"],
                None,
            ],
            ["-", "_", "**", None],
            "rep_str",
            "sep_str",
            ["a-rep_str-b", None, "rep_str**hello**rep_str**world", None],
        ),
        (
            [[None, "a"], [None], None],
            ["-", "_", None],
            "rep_str",
            None,
            ["rep_str-a", None, None],
        ),
        (
            [[None, "a"], [None], None],
            ["-", "_", None],
            None,
            "sep_str",
            ["-a", None, None],
        ),
    ],
)
def test_str_join_lists(sr, sep, string_na_rep, sep_na_rep, expected):
    sr = cudf.Series(sr)
    actual = sr.str.join(
        sep=sep, string_na_rep=string_na_rep, sep_na_rep=sep_na_rep
    )
    expected = cudf.Series(expected)
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "patterns, expected",
    [
        (
            lambda: ["a", "s", "g", "i", "o", "r"],
            [
                [-1, 0, 5, 3, -1, 2],
                [-1, -1, -1, -1, 1, -1],
                [2, 0, -1, -1, -1, 3],
                [-1, -1, -1, 0, -1, -1],
            ],
        ),
        (
            lambda: cudf.Series(["a", "string", "g", "inn", "o", "r", "sea"]),
            [
                [-1, 0, 5, -1, -1, 2, -1],
                [-1, -1, -1, -1, 1, -1, -1],
                [2, -1, -1, -1, -1, 3, 0],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
        ),
    ],
)
def test_str_find_multiple(patterns, expected):
    s = cudf.Series(["strings", "to", "search", "in"])
    t = patterns()

    expected = cudf.Series(expected)

    # We convert to pandas because find_multiple returns ListDtype(int32)
    # and expected is ListDtype(int64).
    # Currently there is no easy way to type-cast these to match.
    assert_eq(s.str.find_multiple(t).to_pandas(), expected.to_pandas())

    s = cudf.Index(s)
    t = cudf.Index(t)

    expected.index = s

    assert_eq(s.str.find_multiple(t).to_pandas(), expected.to_pandas())


def test_str_find_multiple_error():
    s = cudf.Series(["strings", "to", "search", "in"])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "patterns should be an array-like or a Series object, found "
            "<class 'str'>"
        ),
    ):
        s.str.find_multiple("a")

    t = cudf.Series([1, 2, 3])
    with pytest.raises(
        TypeError,
        match=re.escape("patterns can only be of 'string' dtype, got: int64"),
    ):
        s.str.find_multiple(t)


def test_str_iterate_error():
    s = cudf.Series(["abc", "xyz"])
    with pytest.raises(TypeError):
        iter(s.str)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "pqr", "tuv"],
        ["aaaaaaaaaaaa", None],
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        0,
        1,
        2,
        slice(0, 1, 2),
        slice(0, 5, 2),
        slice(-1, -2, 1),
        slice(-1, -2, -1),
        slice(-2, -1, -1),
        slice(-2, -1, 1),
        slice(0),
        slice(None),
    ],
)
def test_string_str_subscriptable(data, index):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    assert_eq(psr.str[index], gsr.str[index])

    psi = pd.Index(data)
    gsi = cudf.Index(data)

    assert_eq(psi.str[index], gsi.str[index])


@pytest.mark.parametrize(
    "data,expected",
    [
        (["aaaaaaaaaaaa"], [12]),
        (["abc", "d", "ef"], [3, 1, 2]),
        (["Hello", "Bye", "Thanks 😊"], [5, 3, 11]),
        (["\n\t", "Bye", "Thanks 😊"], [2, 3, 11]),
    ],
)
def test_string_str_byte_count(data, expected):
    sr = cudf.Series(data)
    expected = cudf.Series(expected, dtype="int32")
    actual = sr.str.byte_count()
    assert_eq(expected, actual)

    si = cudf.Index(data)
    expected = cudf.Index(expected, dtype="int32")
    actual = si.str.byte_count()
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data,expected",
    [
        (["1", "2", "3", "4", "5"], [True, True, True, True, True]),
        (
            ["1.1", "2.0", "3.2", "4.3", "5."],
            [False, False, False, False, False],
        ),
        (
            [".12312", "213123.", ".3223.", "323423.."],
            [False, False, False, False],
        ),
        ([""], [False]),
        (
            ["1..1", "+2", "++3", "4++", "-5"],
            [False, True, False, False, True],
        ),
        (
            [
                "24313345435345 ",
                "+2632726478",
                "++367293674326",
                "4382493264392746.237649274692++",
                "-578239479238469264",
            ],
            [False, True, False, False, True],
        ),
        (
            ["2a2b", "a+b", "++a", "a.b++", "-b"],
            [False, False, False, False, False],
        ),
        (
            ["2a2b", "1+3", "9.0++a", "+", "-"],
            [False, False, False, False, False],
        ),
    ],
)
def test_str_isinteger(data, expected):
    sr = cudf.Series(data, dtype="str")
    expected = cudf.Series(expected)
    actual = sr.str.isinteger()
    assert_eq(expected, actual)

    sr = cudf.Index(data)
    expected = cudf.Index(expected)
    actual = sr.str.isinteger()
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data,expected",
    [
        (["1", "2", "3", "4", "5"], [True, True, True, True, True]),
        (["1.1", "2.0", "3.2", "4.3", "5."], [True, True, True, True, True]),
        ([""], [False]),
        (
            [".12312", "213123.", ".3223.", "323423.."],
            [True, True, False, False],
        ),
        (
            ["1.00.323.1", "+2.1", "++3.30", "4.9991++", "-5.3"],
            [False, True, False, False, True],
        ),
        (
            [
                "24313345435345 ",
                "+2632726478",
                "++367293674326",
                "4382493264392746.237649274692++",
                "-578239479238469264",
            ],
            [False, True, False, False, True],
        ),
        (
            [
                "24313345435345.32732 ",
                "+2632726478.3627638276",
                "++0.326294632367293674326",
                "4382493264392746.237649274692++",
                "-57823947923.8469264",
            ],
            [False, True, False, False, True],
        ),
        (
            ["2a2b", "a+b", "++a", "a.b++", "-b"],
            [False, False, False, False, False],
        ),
        (
            ["2a2b", "1+3", "9.0++a", "+", "-"],
            [False, False, False, False, False],
        ),
    ],
)
def test_str_isfloat(data, expected):
    sr = cudf.Series(data, dtype="str")
    expected = cudf.Series(expected)
    actual = sr.str.isfloat()
    assert_eq(expected, actual)

    sr = cudf.Index(data)
    expected = cudf.Index(expected)
    actual = sr.str.isfloat()
    assert_eq(expected, actual)


def test_string_isipv4():
    gsr = cudf.Series(
        [
            "",
            None,
            "1...1",
            "141.168.0.1",
            "127.0.0.1",
            "1.255.0.1",
            "256.27.28.26",
            "25.257.28.26",
            "25.27.258.26",
            "25.27.28.256",
            "-1.0.0.0",
        ]
    )
    got = gsr.str.isipv4()
    expected = cudf.Series(
        [
            False,
            None,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
    )
    assert_eq(expected, got)


def test_string_ip4_to_int():
    gsr = cudf.Series(
        ["", None, "hello", "41.168.0.1", "127.0.0.1", "41.197.0.1"]
    )
    expected = cudf.Series([0, None, 0, 698875905, 2130706433, 700776449])

    got = gsr.str.ip2int()
    assert_eq(expected, got)

    got = gsr.str.ip_to_int()  # alias
    assert_eq(expected, got)


def test_string_istimestamp():
    gsr = cudf.Series(
        [
            "",
            None,
            "20201009 123456.987654AM+0100",
            "1920111 012345.000001",
            "18201235 012345.1",
            "20201009 250001.2",
            "20201009 129901.3",
            "20201009 123499.4",
            "20201009 000000.500000PM-0130",
            "20201009:000000.600000",
            "20201009 010203.700000PM-2500",
            "20201009 010203.800000AM+0590",
            "20201009 010203.900000AP-0000",
        ]
    )
    got = gsr.str.istimestamp(r"%Y%m%d %H%M%S.%f%p%z")
    expected = cudf.Series(
        [
            False,
            None,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ]
    )
    assert_eq(expected, got)


def test_istimestamp_empty():
    gsr = cudf.Series([], dtype="object")
    result = gsr.str.istimestamp("%Y%m%d")
    expected = cudf.Series([], dtype="bool")
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        ["f0:18:98:22:c2:e4", "00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff"],
        ["f0189822c2e4", "000000000000", "ffffffffffff"],
        ["0xf0189822c2e4", "0x000000000000", "0xffffffffffff"],
        ["0Xf0189822c2e4", "0X000000000000", "0Xffffffffffff"],
    ],
)
def test_string_hex_to_int(data):
    gsr = cudf.Series(data)

    expected = cudf.Series([263988422296292, 0, 281474976710655])

    got = gsr.str.htoi()
    assert_eq(expected, got)

    got = gsr.str.hex_to_int()  # alias
    assert_eq(expected, got)


def test_string_ishex():
    gsr = cudf.Series(["", None, "0x01a2b3c4d5e6f", "0789", "ABCDEF0"])
    got = gsr.str.ishex()
    expected = cudf.Series([False, None, True, True, True])
    assert_eq(expected, got)


def test_string_str_code_points():
    data = [
        "abc",
        "Def",
        None,
        "jLl",
        "dog and cat",
        "accénted",
        "",
        " 1234 ",
        "XYZ",
    ]
    gs = cudf.Series(data)
    expected = [
        97,
        98,
        99,
        68,
        101,
        102,
        106,
        76,
        108,
        100,
        111,
        103,
        32,
        97,
        110,
        100,
        32,
        99,
        97,
        116,
        97,
        99,
        99,
        50089,
        110,
        116,
        101,
        100,
        32,
        49,
        50,
        51,
        52,
        32,
        88,
        89,
        90,
    ]
    expected = cudf.Series(expected)

    assert_eq(expected, gs.str.code_points(), check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        ["http://www.hellow.com", "/home/nvidia/nfs", "123.45 ~ABCDEF"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
    ],
)
def test_string_str_url_encode(data):
    gs = cudf.Series(data)

    got = gs.str.url_encode()
    expected = pd.Series([urllib.parse.quote(url, safe="~") for url in data])
    assert_eq(expected, got)


def test_string_str_decode_url():
    data = [
        "http://www.hellow.com?k1=acc%C3%A9nted&k2=a%2F/b.c",
        "%2Fhome%2fnfs",
        "987%20ZYX",
    ]
    gs = cudf.Series(data)

    got = gs.str.url_decode()
    expected = pd.Series([urllib.parse.unquote(url) for url in data])
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
        ["line to be wrapped", "another line to be wrapped"],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
def test_string_str_translate(data):
    ps = pd.Series(data)
    gs = cudf.Series(data)

    assert_eq(
        ps.str.translate(str.maketrans({"a": "z"})),
        gs.str.translate(str.maketrans({"a": "z"})),
    )
    assert_eq(
        pd.Index(ps).str.translate(str.maketrans({"a": "z"})),
        cudf.Index(gs).str.translate(str.maketrans({"a": "z"})),
    )
    assert_eq(
        ps.str.translate(str.maketrans({"a": "z", "i": "$", "z": "1"})),
        gs.str.translate(str.maketrans({"a": "z", "i": "$", "z": "1"})),
    )
    assert_eq(
        pd.Index(ps).str.translate(
            str.maketrans({"a": "z", "i": "$", "z": "1"})
        ),
        cudf.Index(gs).str.translate(
            str.maketrans({"a": "z", "i": "$", "z": "1"})
        ),
    )
    assert_eq(
        ps.str.translate(
            str.maketrans({"+": "-", "-": "$", "?": "!", "B": "."})
        ),
        gs.str.translate(
            str.maketrans({"+": "-", "-": "$", "?": "!", "B": "."})
        ),
    )
    assert_eq(
        pd.Index(ps).str.translate(
            str.maketrans({"+": "-", "-": "$", "?": "!", "B": "."})
        ),
        cudf.Index(gs).str.translate(
            str.maketrans({"+": "-", "-": "$", "?": "!", "B": "."})
        ),
    )
    assert_eq(
        ps.str.translate(str.maketrans({"é": "É"})),
        gs.str.translate(str.maketrans({"é": "É"})),
    )


def test_string_str_filter_characters():
    data = [
        "hello world",
        "A+B+C+D",
        "?!@#$%^&*()",
        "accént",
        None,
        "$1.50",
        "",
    ]
    gs = cudf.Series(data)
    expected = cudf.Series(
        ["helloworld", "ABCD", "", "accnt", None, "150", ""]
    )
    filter = {"a": "z", "A": "Z", "0": "9"}
    assert_eq(expected, gs.str.filter_characters(filter))

    expected = cudf.Series([" ", "+++", "?!@#$%^&*()", "é", None, "$.", ""])
    assert_eq(expected, gs.str.filter_characters(filter, False))

    expected = cudf.Series(
        ["hello world", "A B C D", "           ", "acc nt", None, " 1 50", ""]
    )
    assert_eq(expected, gs.str.filter_characters(filter, True, " "))

    with pytest.raises(TypeError):
        gs.str.filter_characters(filter, True, ["a"])


@pytest.mark.parametrize(
    "data,sub,er",
    [
        (["abc", "xyz", "a", "ab", "123", "097"], "a", ValueError),
        (["A B", "1.5", "3,000"], "abc", ValueError),
        (["23", "³", "⅕", ""], "⅕", ValueError),
        ([" ", "\t\r\n ", ""], "\n", ValueError),
        (["$", "B", "Aab$", "$$ca", "C$B$", "cat"], "$", ValueError),
        (["line to be wrapped", "another line to be wrapped"], " ", None),
        (
            ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
            "+",
            ValueError,
        ),
        (["line to be wrapped", "another line to be wrapped"], "", None),
    ],
)
def test_string_str_rindex(data, sub, er):
    ps = pd.Series(data)
    gs = cudf.Series(data)

    if er is None:
        assert_eq(ps.str.rindex(sub), gs.str.rindex(sub), check_dtype=False)
        assert_eq(
            pd.Index(ps).str.rindex(sub),
            cudf.Index(gs).str.rindex(sub),
            exact=False,
        )

    try:
        ps.str.rindex(sub)
    except er:
        pass
    else:
        assert not er

    try:
        gs.str.rindex(sub)
    except er:
        pass
    else:
        assert not er


@pytest.mark.parametrize(
    "data,sub,expect",
    [
        (
            ["abc", "xyz", "a", "ab", "123", "097"],
            ["b", "y", "a", "c", "4", "8"],
            [True, True, True, False, False, False],
        ),
        (
            ["A B", "1.5", "3,000", "23", "³", "⅕"],
            ["A B", ".", ",", "1", " ", " "],
            [True, True, True, False, False, False],
        ),
        (
            [" ", "\t", "\r", "\f ", "\n", ""],
            ["", "\t", "\r", "xx", "yy", "zz"],
            [True, True, True, False, False, False],
        ),
        (
            ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
            ["$", "B", "ab", "*", "@", "dog"],
            [True, True, True, False, False, False],
        ),
        (
            ["hello", "there", "world", "-1234", None, "accént"],
            ["lo", "e", "o", "+1234", " ", "e"],
            [True, True, True, False, None, False],
        ),
        (
            ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", "", "x", None],
            ["A", "B", "C", " ", "y", "e"],
            [True, True, True, False, False, None],
        ),
    ],
)
def test_string_contains_multi(data, sub, expect):
    gs = cudf.Series(data)
    sub = cudf.Series(sub)
    got = gs.str.contains(sub)
    expect = cudf.Series(expect)
    assert_eq(expect, got, check_dtype=False)


# Pandas does not allow 'case' or 'flags' if 'pat' is re.Pattern
# This covers contains, match, count, and replace
@pytest.mark.parametrize(
    "pat",
    [re.compile("[n-z]"), re.compile("[A-Z]"), re.compile("de"), "A"],
)
@pytest.mark.parametrize("repl", ["xyz", "", " "])
def test_string_compiled_re(ps_gs, pat, repl):
    ps, gs = ps_gs

    expect = ps.str.contains(pat, regex=True)
    got = gs.str.contains(pat, regex=True)
    assert_eq(expect, got)

    expect = ps.str.match(pat)
    got = gs.str.match(pat)
    assert_eq(expect, got)

    expect = ps.str.count(pat)
    got = gs.str.count(pat)
    assert_eq(expect, got, check_dtype=False)

    expect = ps.str.replace(pat, repl, regex=True)
    got = gs.str.replace(pat, repl, regex=True)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
        ["line to be wrapped", "another line to be wrapped"],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize("pat", ["", " ", "a", "abc", "cat", "$", "\n"])
def test_string_str_match(data, pat):
    ps = pd.Series(data)
    gs = cudf.Series(data)

    assert_eq(ps.str.match(pat), gs.str.match(pat))
    assert_eq(
        pd.Index(pd.Index(ps).str.match(pat)), cudf.Index(gs).str.match(pat)
    )


@pytest.mark.parametrize(
    "data,sub,er",
    [
        (["abc", "xyz", "a", "ab", "123", "097"], "a", ValueError),
        (["A B", "1.5", "3,000"], "abc", ValueError),
        (["23", "³", "⅕", ""], "⅕", ValueError),
        ([" ", "\t\r\n ", ""], "\n", ValueError),
        (["$", "B", "Aab$", "$$ca", "C$B$", "cat"], "$", ValueError),
        (["line to be wrapped", "another line to be wrapped"], " ", None),
        (
            ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
            "+",
            ValueError,
        ),
        (["line to be wrapped", "another line to be wrapped"], "", None),
    ],
)
def test_string_str_index(data, sub, er):
    ps = pd.Series(data)
    gs = cudf.Series(data)

    if er is None:
        assert_eq(ps.str.index(sub), gs.str.index(sub), check_dtype=False)

    try:
        ps.str.index(sub)
    except er:
        pass
    else:
        assert not er

    try:
        gs.str.index(sub)
    except er:
        pass
    else:
        assert not er


@pytest.mark.parametrize(
    "data",
    [
        ["str_foo", "str_bar", "no_prefix", "", None],
        ["foo_str", "bar_str", "no_suffix", "", None],
    ],
)
def test_string_remove_suffix_prefix(data):
    ps = pd.Series(data)
    gs = cudf.Series(data)

    got = gs.str.removeprefix("str_")
    expect = ps.str.removeprefix("str_")
    assert_eq(
        expect,
        got,
        check_dtype=False,
    )
    got = gs.str.removesuffix("_str")
    expect = ps.str.removesuffix("_str")
    assert_eq(
        expect,
        got,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
        ["line to be wrapped", "another line to be wrapped"],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize(
    "sub",
    ["", " ", "a", "abc", "cat", "$", "\n"],
)
def test_string_find(data, sub):
    ps = pd.Series(data)
    gs = cudf.Series(data)

    got = gs.str.find(sub)
    expect = ps.str.find(sub)
    assert_eq(
        expect,
        got,
        check_dtype=False,
    )

    got = gs.str.find(sub, start=1)
    expect = ps.str.find(sub, start=1)
    assert_eq(
        expect,
        got,
        check_dtype=False,
    )

    got = gs.str.find(sub, end=10)
    expect = ps.str.find(sub, end=10)
    assert_eq(
        expect,
        got,
        check_dtype=False,
    )

    got = gs.str.find(sub, start=2, end=10)
    expect = ps.str.find(sub, start=2, end=10)
    assert_eq(
        expect,
        got,
        check_dtype=False,
    )

    got = gs.str.rfind(sub)
    expect = ps.str.rfind(sub)
    assert_eq(
        expect,
        got,
        check_dtype=False,
    )

    got = gs.str.rfind(sub, start=1)
    expect = ps.str.rfind(sub, start=1)
    assert_eq(
        expect,
        got,
        check_dtype=False,
    )

    got = gs.str.rfind(sub, end=10)
    expect = ps.str.rfind(sub, end=10)
    assert_eq(
        expect,
        got,
        check_dtype=False,
    )

    got = gs.str.rfind(sub, start=2, end=10)
    expect = ps.str.rfind(sub, start=2, end=10)
    assert_eq(
        expect,
        got,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
        ["line to be wrapped", "another line to be wrapped"],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize(
    "pat",
    ["", None, " ", "a", "abc", "cat", "$", "\n"],
)
def test_string_starts_ends(data, pat):
    ps = pd.Series(data)
    gs = cudf.Series(data)

    if pat is None:
        assert_exceptions_equal(
            lfunc=ps.str.startswith,
            rfunc=gs.str.startswith,
            lfunc_args_and_kwargs=([pat],),
            rfunc_args_and_kwargs=([pat],),
        )
        assert_exceptions_equal(
            lfunc=ps.str.endswith,
            rfunc=gs.str.endswith,
            lfunc_args_and_kwargs=([pat],),
            rfunc_args_and_kwargs=([pat],),
        )
    else:
        assert_eq(
            ps.str.startswith(pat), gs.str.startswith(pat), check_dtype=False
        )
        assert_eq(
            ps.str.endswith(pat), gs.str.endswith(pat), check_dtype=False
        )


@pytest.mark.parametrize(
    "data,pat",
    [
        (
            ["abc", "xyz", "a", "ab", "123", "097"],
            ("abc", "x", "a", "b", "3", "7"),
        ),
        (["A B", "1.5", "3,000"], ("A ", ".", ",")),
        (["23", "³", "⅕", ""], ("23", "³", "⅕", "")),
        ([" ", "\t\r\n ", ""], ("d", "\n ", "")),
        (
            ["$", "B", "Aab$", "$$ca", "C$B$", "cat"],
            ("$", "$", "a", "<", "(", "#"),
        ),
        (
            ["line to be wrapped", "another line to be wrapped"],
            ("another", "wrapped"),
        ),
        (
            ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
            ("hsdjfk", "", "ll", "+", "-", "w", "-", "én"),
        ),
        (
            ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
            ("1. Ant.  ", "2. Bee!\n", "3. Cat?\t", ""),
        ),
    ],
)
def test_string_starts_ends_list_like_pat(data, pat):
    gs = cudf.Series(data)

    starts_expected = []
    ends_expected = []
    for i in range(len(pat)):
        if data[i] is None:
            starts_expected.append(None)
            ends_expected.append(None)
        else:
            if pat[i] is None:
                starts_expected.append(False)
                ends_expected.append(False)
            else:
                starts_expected.append(data[i].startswith(pat[i]))
                ends_expected.append(data[i].endswith(pat[i]))
    starts_expected = pd.Series(starts_expected)
    ends_expected = pd.Series(ends_expected)
    assert_eq(starts_expected, gs.str.startswith(pat), check_dtype=False)
    assert_eq(ends_expected, gs.str.endswith(pat), check_dtype=False)


@pytest.mark.parametrize(
    "find",
    [
        "(\\d)(\\d)",
        "(\\d)(\\d)",
        "(\\d)(\\d)",
        "(\\d)(\\d)",
        "([a-z])-([a-z])",
        "([a-z])-([a-zé])",
        "([a-z])-([a-z])",
        "([a-z])-([a-zé])",
        re.compile("([A-Z])(\\d)"),
    ],
)
@pytest.mark.parametrize(
    "replace",
    ["\\1-\\2", "V\\2-\\1", "\\1 \\2", "\\2 \\1", "X\\1+\\2Z", "X\\1+\\2Z"],
)
def test_string_replace_with_backrefs(find, replace):
    s = [
        "A543",
        "Z756",
        "",
        None,
        "tést-string",
        "two-thréé four-fivé",
        "abcd-éfgh",
        "tést-string-again",
    ]
    ps = pd.Series(s)
    gs = cudf.Series(s)
    got = gs.str.replace_with_backrefs(find, replace)
    expected = ps.str.replace(find, replace, regex=True)
    assert_eq(got, expected)

    got = cudf.Index(gs).str.replace_with_backrefs(find, replace)
    expected = pd.Index(ps).str.replace(find, replace, regex=True)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["$", "B", "Aab$", "$$ca", "C$B$", "cat", "cat\ndog"],
        ["line\nto be wrapped", "another\nline\nto be wrapped"],
    ],
)
@pytest.mark.parametrize(
    "pat",
    ["a", " ", "\t", "another", "0", r"\$", "^line$", "line.*be", "cat$"],
)
@pytest.mark.parametrize("flags", [0, re.MULTILINE, re.DOTALL])
def test_string_count(data, pat, flags):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(
        gs.str.count(pat=pat, flags=flags),
        ps.str.count(pat=pat, flags=flags),
        check_dtype=False,
    )
    assert_eq(
        cudf.Index(gs).str.count(pat=pat),
        pd.Index(ps).str.count(pat=pat),
        exact=False,
    )


@pytest.mark.parametrize(
    "pat, flags",
    [
        ("Monkey", 0),
        ("on", 0),
        ("b", 0),
        ("on$", 0),
        ("on$", re.MULTILINE),
        ("o.*k", re.DOTALL),
    ],
)
def test_string_findall(pat, flags):
    test_data = ["Lion", "Monkey", "Rabbit", "Don\nkey"]
    ps = pd.Series(test_data)
    gs = cudf.Series(test_data)

    expected = ps.str.findall(pat, flags)
    actual = gs.str.findall(pat, flags)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "pat, flags, pos",
    [
        ("Monkey", 0, [-1, 0, -1, -1]),
        ("on", 0, [2, 1, -1, 1]),
        ("bit", 0, [-1, -1, 3, -1]),
        ("on$", 0, [2, -1, -1, -1]),
        ("on$", re.MULTILINE, [2, -1, -1, 1]),
        ("o.*k", re.DOTALL, [-1, 1, -1, 1]),
    ],
)
def test_string_find_re(pat, flags, pos):
    test_data = ["Lion", "Monkey", "Rabbit", "Don\nkey"]
    gs = cudf.Series(test_data)

    expected = pd.Series(pos, dtype=np.int32)
    actual = gs.str.find_re(pat, flags)
    assert_eq(expected, actual)


def test_string_replace_multi():
    ps = pd.Series(["hello", "goodbye"])
    gs = cudf.Series(["hello", "goodbye"])
    expect = ps.str.replace("e", "E").str.replace("o", "O")
    got = gs.str.replace(["e", "o"], ["E", "O"])

    assert_eq(expect, got)

    ps = pd.Series(["foo", "fuz", np.nan])
    gs = cudf.Series.from_pandas(ps)

    expect = ps.str.replace("f.", "ba", regex=True)
    got = gs.str.replace(["f."], ["ba"], regex=True)
    assert_eq(expect, got)

    ps = pd.Series(["f.o", "fuz", np.nan])
    gs = cudf.Series.from_pandas(ps)

    expect = ps.str.replace("f.", "ba", regex=False)
    got = gs.str.replace(["f."], ["ba"], regex=False)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["+23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize("width", [0, 1, 25])
@pytest.mark.parametrize("side", ["left", "right", "both"])
@pytest.mark.parametrize("fillchar", [" ", ".", "\n", "+", "\t"])
def test_strings_pad_tests(data, width, side, fillchar):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(
        ps.str.pad(width=width, side=side, fillchar=fillchar),
        gs.str.pad(width=width, side=side, fillchar=fillchar),
    )

    gi = cudf.Index(data)
    pi = pd.Index(data)

    assert_eq(
        pi.str.pad(width=width, side=side, fillchar=fillchar),
        gi.str.pad(width=width, side=side, fillchar=fillchar),
    )


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        pytest.param([" ", "\t\r\n ", ""], marks=pytest.mark.xfail),
        ["leopard", "Golden Eagle", "SNAKE", ""],
        ["line to be wrapped", "another line to be wrapped"],
    ],
)
@pytest.mark.parametrize("width", [1, 20])
def test_string_wrap(data, width):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(
        gs.str.wrap(
            width=width,
            break_long_words=False,
            expand_tabs=False,
            replace_whitespace=True,
            drop_whitespace=True,
            break_on_hyphens=False,
        ),
        ps.str.wrap(
            width=width,
            break_long_words=False,
            expand_tabs=False,
            replace_whitespace=True,
            drop_whitespace=True,
            break_on_hyphens=False,
        ),
    )

    gi = cudf.Index(data)
    pi = pd.Index(data)

    assert_eq(
        gi.str.wrap(
            width=width,
            break_long_words=False,
            expand_tabs=False,
            replace_whitespace=True,
            drop_whitespace=True,
            break_on_hyphens=False,
        ),
        pi.str.wrap(
            width=width,
            break_long_words=False,
            expand_tabs=False,
            replace_whitespace=True,
            drop_whitespace=True,
            break_on_hyphens=False,
        ),
    )


@pytest.mark.parametrize(
    "data",
    [
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["³", "⅕", ""],
        ["hello", "there", "world", "+1234", "-1234", None, "accént", ""],
        [" ", "\t\r\n ", ""],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize("width", [0, 20])
def test_strings_zfill_tests(data, width):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(ps.str.zfill(width=width), gs.str.zfill(width=width))

    gi = cudf.Index(data)
    pi = pd.Index(data)

    assert_eq(pi.str.zfill(width=width), gi.str.zfill(width=width))


def test_string_strip_fail():
    gs = cudf.Series(["a", "aa", ""])
    with pytest.raises(TypeError):
        gs.str.strip(["a"])
    with pytest.raises(TypeError):
        gs.str.lstrip(["a"])
    with pytest.raises(TypeError):
        gs.str.rstrip(["a"])


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        [
            "this is a regular sentence",
            "https://docs.python.org/3/tutorial/index.html",
            None,
        ],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize("width", [0, 20])
@pytest.mark.parametrize("fillchar", ["⅕", "1", ".", "t", " ", ","])
def test_strings_filling_tests(data, width, fillchar):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(
        ps.str.center(width=width, fillchar=fillchar),
        gs.str.center(width=width, fillchar=fillchar),
    )
    assert_eq(
        ps.str.ljust(width=width, fillchar=fillchar),
        gs.str.ljust(width=width, fillchar=fillchar),
    )
    assert_eq(
        ps.str.rjust(width=width, fillchar=fillchar),
        gs.str.rjust(width=width, fillchar=fillchar),
    )

    gi = cudf.Index(data)
    pi = pd.Index(data)

    assert_eq(
        pi.str.center(width=width, fillchar=fillchar),
        gi.str.center(width=width, fillchar=fillchar),
    )
    assert_eq(
        pi.str.ljust(width=width, fillchar=fillchar),
        gi.str.ljust(width=width, fillchar=fillchar),
    )
    assert_eq(
        pi.str.rjust(width=width, fillchar=fillchar),
        gi.str.rjust(width=width, fillchar=fillchar),
    )


@pytest.mark.parametrize("n", [-1, 0, 1, 4])
@pytest.mark.parametrize("expand", [True, False])
def test_string_rsplit_re(n, expand):
    data = ["a b", " c ", "   d", "e   ", "f"]
    ps = pd.Series(data, dtype="str")
    gs = cudf.Series(data, dtype="str")

    # Pandas does not yet support the regex parameter for rsplit
    import inspect

    assert (
        "regex"
        not in inspect.signature(pd.Series.str.rsplit).parameters.keys()
    )

    expect = ps.str.rsplit(pat=" ", n=n, expand=expand)
    got = gs.str.rsplit(pat="\\s", n=n, expand=expand, regex=True)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        [
            "this is a regular sentence",
            "https://docs.python.org/3/tutorial/index.html",
            None,
        ],
    ],
)
@pytest.mark.parametrize("n", [-1, 0, 1, 4])
@pytest.mark.parametrize("expand", [True, False])
def test_strings_split(data, n, expand):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(
        ps.str.split(n=n, expand=expand).reset_index(),
        gs.str.split(n=n, expand=expand).reset_index(),
        check_index_type=False,
    )

    assert_eq(
        ps.str.split(",", n=n, expand=expand),
        gs.str.split(",", n=n, expand=expand),
    )
    assert_eq(
        ps.str.split("-", n=n, expand=expand),
        gs.str.split("-", n=n, expand=expand),
    )


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        [
            "this is a regular sentence",
            "https://docs.python.org/3/tutorial/index.html",
            None,
        ],
        ["1. Ant.  ", "2. Bee!\n", "3. Cat?\t", None],
    ],
)
@pytest.mark.parametrize(
    "to_strip", ["⅕", None, "123.", ".!? \n\t", "123.!? \n\t", " ", ".", ","]
)
def test_strings_strip_tests(data, to_strip):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(ps.str.strip(to_strip=to_strip), gs.str.strip(to_strip=to_strip))
    assert_eq(
        ps.str.rstrip(to_strip=to_strip), gs.str.rstrip(to_strip=to_strip)
    )
    assert_eq(
        ps.str.lstrip(to_strip=to_strip), gs.str.lstrip(to_strip=to_strip)
    )

    gi = cudf.Index(data)
    pi = pd.Index(data)

    assert_eq(pi.str.strip(to_strip=to_strip), gi.str.strip(to_strip=to_strip))
    assert_eq(
        pi.str.rstrip(to_strip=to_strip), gi.str.rstrip(to_strip=to_strip)
    )
    assert_eq(
        pi.str.lstrip(to_strip=to_strip), gi.str.lstrip(to_strip=to_strip)
    )


def test_string_is_title():
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
    ]
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(gs.str.istitle(), ps.str.istitle())


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
    ],
)
def test_strings_rpartition(data):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(ps.str.rpartition(), gs.str.rpartition())
    assert_eq(ps.str.rpartition("-"), gs.str.rpartition("-"))
    assert_eq(ps.str.rpartition(","), gs.str.rpartition(","))


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
    ],
)
def test_strings_partition(data):
    gs = cudf.Series(data, name="str_name")
    ps = pd.Series(data, name="str_name")

    assert_eq(ps.str.partition(), gs.str.partition())
    assert_eq(ps.str.partition(","), gs.str.partition(","))
    assert_eq(ps.str.partition("-"), gs.str.partition("-"))

    gi = cudf.Index(data, name="new name")
    pi = pd.Index(data, name="new name")
    assert_eq(pi.str.partition(), gi.str.partition())
    assert_eq(pi.str.partition(","), gi.str.partition(","))
    assert_eq(pi.str.partition("-"), gi.str.partition("-"))


def test_string_partition_fail():
    gs = cudf.Series(["abc", "aa", "cba"])
    with pytest.raises(TypeError):
        gs.str.partition(["a"])
    with pytest.raises(TypeError):
        gs.str.rpartition(["a"])


@pytest.mark.parametrize(
    "data",
    [
        ["koala", "fox", "chameleon"],
        ["A,,B", "1,,5", "3,00,0"],
        ["Linda van der Berg", "George Pitt-Rivers"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        [
            "this is a regular sentence",
            "https://docs.python.org/3/tutorial/index.html",
            None,
        ],
    ],
)
@pytest.mark.parametrize("n", [-1, 2, 9])
@pytest.mark.parametrize("expand", [True, False])
def test_strings_rsplit(data, n, expand):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(
        ps.str.rsplit(n=n, expand=expand).reset_index(),
        gs.str.rsplit(n=n, expand=expand).reset_index(),
        check_index_type=False,
    )
    assert_eq(
        ps.str.rsplit(",", n=n, expand=expand),
        gs.str.rsplit(",", n=n, expand=expand),
    )
    assert_eq(
        ps.str.rsplit("-", n=n, expand=expand),
        gs.str.rsplit("-", n=n, expand=expand),
    )


@pytest.fixture(
    params=[
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""],
        ["koala", "fox", "chameleon"],
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
        ],
        ["one", "one1", "1", ""],
        ["A B", "1.5", "3,000"],
        ["23", "³", "⅕", ""],
        [" ", "\t\r\n ", ""],
        ["leopard", "Golden Eagle", "SNAKE", ""],
        [r"¯\_(ツ)_/¯", "(╯°□°)╯︵ ┻━┻", "┬─┬ノ( º _ ºノ)"],
        ["a1", "A1", "a!", "A!", "!1", "aA"],
        [
            None,
            "The quick bRoWn fox juMps over the laze DOG",
            '123nr98nv9rev!$#INF4390v03n1243<>?}{:-"',
            "accénted",
        ],
    ]
)
def data_char_types(request):
    return request.param


@pytest.mark.parametrize(
    "type_op",
    [
        "isdecimal",
        "isalnum",
        "isalpha",
        "isdigit",
        "isnumeric",
        "isupper",
        "islower",
    ],
)
def test_string_char_types(type_op, data_char_types):
    gs = cudf.Series(data_char_types)
    ps = pd.Series(data_char_types)

    assert_eq(getattr(gs.str, type_op)(), getattr(ps.str, type_op)())


def test_string_filter_alphanum():
    data = ["1234567890", "!@#$%^&*()", ",./<>?;:[]}{|+=", "abc DEF"]
    expected = []
    for st in data:
        rs = ""
        for c in st:
            if str.isalnum(c):
                rs = rs + c
        expected.append(rs)

    gs = cudf.Series(data)
    assert_eq(gs.str.filter_alphanum(), cudf.Series(expected))

    expected = []
    for st in data:
        rs = ""
        for c in st:
            if not str.isalnum(c):
                rs = rs + c
        expected.append(rs)
    assert_eq(gs.str.filter_alphanum(keep=False), cudf.Series(expected))

    expected = []
    for st in data:
        rs = ""
        for c in st:
            if str.isalnum(c):
                rs = rs + c
            else:
                rs = rs + "*"
        expected.append(rs)
    assert_eq(gs.str.filter_alphanum("*"), cudf.Series(expected))

    expected = []
    for st in data:
        rs = ""
        for c in st:
            if not str.isalnum(c):
                rs = rs + c
            else:
                rs = rs + "*"
        expected.append(rs)
    assert_eq(gs.str.filter_alphanum("*", keep=False), cudf.Series(expected))

    with pytest.raises(TypeError):
        gs.str.filter_alphanum(["a"])


@pytest.mark.parametrize(
    "case_op",
    [
        "title",
        "capitalize",
        "lower",
        "upper",
        "swapcase",
        "isdecimal",
        "isalnum",
        "isalpha",
        "isdigit",
        "isnumeric",
        "isspace",
    ],
)
def test_string_char_case(case_op, data_char_types):
    gs = cudf.Series(data_char_types)
    ps = pd.Series(data_char_types)
    assert_eq(getattr(gs.str, case_op)(), getattr(ps.str, case_op)())


def test_string_isempty(data_char_types):
    gs = cudf.Series(data_char_types)
    ps = pd.Series(data_char_types)
    assert_eq(gs.str.isempty(), ps == "")


@pytest.mark.parametrize(
    "string",
    [
        ["Cbe", "cbe", "CbeD", "Cb", "ghi", "Cb"],
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""],
    ],
)
@pytest.mark.parametrize("index", [-100, -3, -1, 0, 1, 4, 50])
def test_string_get(string, index):
    pds = pd.Series(string)
    gds = cudf.Series(string)

    assert_eq(
        pds.str.get(index).fillna(""),
        gds.str.get(index).fillna(""),
    )


@pytest.mark.parametrize(
    "string",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""],
        ["koala", "fox", "chameleon"],
    ],
)
@pytest.mark.parametrize("number", [-10, 0, 1, 3, 10])
@pytest.mark.parametrize("diff", [0, 3])
def test_string_slice_str(string, number, diff):
    pds = pd.Series(string)
    gds = cudf.Series(string)

    assert_eq(pds.str.slice(start=number), gds.str.slice(start=number))
    assert_eq(pds.str.slice(stop=number), gds.str.slice(stop=number))
    assert_eq(pds.str.slice(), gds.str.slice())
    assert_eq(
        pds.str.slice(start=number, stop=number + diff),
        gds.str.slice(start=number, stop=number + diff),
    )
    if diff != 0:
        assert_eq(pds.str.slice(step=diff), gds.str.slice(step=diff))
        assert_eq(
            pds.str.slice(start=number, stop=number + diff, step=diff),
            gds.str.slice(start=number, stop=number + diff, step=diff),
        )


def test_string_slice_from():
    gs = cudf.Series(["hello world", "holy accéntéd", "batman", None, ""])
    d_starts = cudf.Series([2, 3, 0, -1, -1], dtype=np.int32)
    d_stops = cudf.Series([-1, -1, 0, -1, -1], dtype=np.int32)
    got = gs.str.slice_from(starts=d_starts, stops=d_stops)
    expected = cudf.Series(["llo world", "y accéntéd", "", None, ""])
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "string",
    [
        ["abc", "xyz", "a", "ab", "123", "097"],
        ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""],
        ["koala", "fox", "chameleon"],
    ],
)
@pytest.mark.parametrize("number", [0, 1, 10])
@pytest.mark.parametrize("diff", [0, 3])
@pytest.mark.parametrize("repl", ["2", "!!"])
def test_string_slice_replace(string, number, diff, repl):
    pds = pd.Series(string)
    gds = cudf.Series(string)

    assert_eq(
        pds.str.slice_replace(start=number, repl=repl),
        gds.str.slice_replace(start=number, repl=repl),
        check_dtype=False,
    )
    assert_eq(
        pds.str.slice_replace(stop=number, repl=repl),
        gds.str.slice_replace(stop=number, repl=repl),
    )
    assert_eq(pds.str.slice_replace(), gds.str.slice_replace())
    assert_eq(
        pds.str.slice_replace(start=number, stop=number + diff),
        gds.str.slice_replace(start=number, stop=number + diff),
    )
    assert_eq(
        pds.str.slice_replace(start=number, stop=number + diff, repl=repl),
        gds.str.slice_replace(start=number, stop=number + diff, repl=repl),
        check_dtype=False,
    )


def test_string_slice_replace_fail():
    gs = cudf.Series(["abc", "xyz", ""])
    with pytest.raises(TypeError):
        gs.str.slice_replace(0, 1, ["_"])


def test_string_insert():
    gs = cudf.Series(["hello world", "holy accéntéd", "batman", None, ""])

    ps = pd.Series(["hello world", "holy accéntéd", "batman", None, ""])

    assert_eq(gs.str.insert(0, ""), gs)
    assert_eq(gs.str.insert(0, "+"), "+" + ps)
    assert_eq(gs.str.insert(-1, "---"), ps + "---")
    assert_eq(
        gs.str.insert(5, "---"),
        ps.str.slice(stop=5) + "---" + ps.str.slice(start=5),
    )

    with pytest.raises(TypeError):
        gs.str.insert(0, ["+"])


def test_string_slice():
    df = cudf.DataFrame({"a": ["hello", "world"]})
    pdf = pd.DataFrame({"a": ["hello", "world"]})
    a_slice_got = df.a.str.slice(0, 2)
    a_slice_expected = pdf.a.str.slice(0, 2)

    assert isinstance(a_slice_got, cudf.Series)
    assert_eq(a_slice_expected, a_slice_got)


@pytest.mark.parametrize("pat", [None, "\\s+"])
@pytest.mark.parametrize("regex", [False, True])
@pytest.mark.parametrize("expand", [False, True])
def test_string_split_all_empty(pat, regex, expand):
    ps = pd.Series(["", "", "", ""], dtype="str")
    gs = cudf.Series(["", "", "", ""], dtype="str")

    expect = ps.str.split(pat=pat, expand=expand, regex=regex)
    got = gs.str.split(pat=pat, expand=expand, regex=regex)

    if isinstance(got, cudf.DataFrame):
        assert_eq(expect, got, check_column_type=False)
    else:
        assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        ["a b", " c ", "   d", "e   ", "f"],
        ["a-b", "-c-", "---d", "e---", "f"],
        ["ab", "c", "d", "e", "f"],
        [None, None, None, None, None],
    ],
)
@pytest.mark.parametrize("pat", [None, " ", "\\-+", "\\s+"])
@pytest.mark.parametrize("n", [-1, 0, 1, 3, 10])
@pytest.mark.parametrize("expand", [True, False])
def test_string_split_re(data, pat, n, expand):
    ps = pd.Series(data, dtype="str")
    gs = cudf.Series(data, dtype="str")

    expect = ps.str.split(pat=pat, n=n, expand=expand, regex=True)
    got = gs.str.split(pat=pat, n=n, expand=expand, regex=True)

    assert_eq(expect, got)


def test_string_lower(ps_gs):
    ps, gs = ps_gs

    expect = ps.str.lower()
    got = gs.str.lower()

    assert_eq(expect, got)


def test_string_upper(ps_gs):
    ps, gs = ps_gs

    expect = ps.str.upper()
    got = gs.str.upper()

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        ["a b", " c ", "   d", "e   ", "f"],
        ["a-b", "-c-", "---d", "e---", "f"],
        ["ab", "c", "d", "e", "f"],
        [None, None, None, None, None],
    ],
)
@pytest.mark.parametrize("pat", [None, " ", "-"])
@pytest.mark.parametrize("n", [-1, 0, 1, 3, 10])
@pytest.mark.parametrize("expand", [True, False])
def test_string_split(data, pat, n, expand):
    ps = pd.Series(data, dtype="str")
    gs = cudf.Series(data, dtype="str")

    expect = ps.str.split(pat=pat, n=n, expand=expand)
    got = gs.str.split(pat=pat, n=n, expand=expand)

    assert_eq(expect, got)


# Pandas doesn't respect the `n` parameter so ignoring it in test parameters
@pytest.mark.parametrize(
    "pat,regex",
    [("a", False), ("f", False), (r"[a-z]", True), (r"[A-Z]", True)],
)
@pytest.mark.parametrize("repl", ["qwerty", "", " "])
@pytest.mark.parametrize("case,case_raise", [(None, 0), (True, 1), (False, 1)])
@pytest.mark.parametrize("flags,flags_raise", [(0, 0), (re.U, 1)])
def test_string_replace(
    ps_gs, pat, repl, case, case_raise, flags, flags_raise, regex
):
    ps, gs = ps_gs

    expectation = raise_builder([case_raise, flags_raise], NotImplementedError)

    with expectation:
        expect = ps.str.replace(pat, repl, case=case, flags=flags, regex=regex)
        got = gs.str.replace(pat, repl, case=case, flags=flags, regex=regex)

        assert_eq(expect, got)


@pytest.mark.parametrize("pat", ["A*", "F?H?"])
def test_string_replace_zero_length(ps_gs, pat):
    ps, gs = ps_gs

    expect = ps.str.replace(pat, "_", regex=True)
    got = gs.str.replace(pat, "_", regex=True)

    assert_eq(expect, got)


@pytest.mark.parametrize("n", [-1, 0, 1])
def test_string_replace_n(n):
    data = ["a,b,c", "d,e,f,g"]
    expect = pd.Series(data).str.replace(pat=",", repl="_", n=n)
    got = cudf.Series(data).str.replace(pat=",", repl="_", n=n)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "pat,regex",
    [
        ("a", False),
        ("a", True),
        ("f", False),
        (r"[a-z]", True),
        (r"[A-Z]", True),
        ("hello", False),
        ("FGHI", False),
    ],
)
@pytest.mark.parametrize(
    "flags,flags_raise",
    [(0, 0), (re.MULTILINE | re.DOTALL, 0), (re.I, 1), (re.I | re.DOTALL, 1)],
)
@pytest.mark.parametrize("na,na_raise", [(np.nan, 0), (None, 1), ("", 1)])
def test_string_contains(ps_gs, pat, regex, flags, flags_raise, na, na_raise):
    ps, gs = ps_gs

    expectation = does_not_raise()
    if flags_raise or na_raise:
        expectation = pytest.raises(NotImplementedError)

    with expectation:
        with expect_warning_if(
            na == "" or (na is None and not (flags_raise or na_raise)),
            match=(
                "Allowing a non-bool 'na' in obj.str.contains is deprecated "
                "and will raise in a future version."
            ),
        ):
            expect = ps.str.contains(pat, flags=flags, na=na, regex=regex)
            got = gs.str.contains(pat, flags=flags, na=na, regex=regex)
        assert_eq(expect, got)


def test_string_contains_case(ps_gs):
    ps, gs = ps_gs
    with pytest.raises(NotImplementedError):
        gs.str.contains("A", case=False)
    expected = ps.str.contains("A", regex=False, case=False)
    got = gs.str.contains("A", regex=False, case=False)
    assert_eq(expected, got)
    got = gs.str.contains("a", regex=False, case=False)
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "pat,esc,expect",
    [
        ("abc", "", [True, False, False, False, False, False]),
        ("b%", "/", [False, True, False, False, False, False]),
        ("%b", ":", [False, True, False, False, False, False]),
        ("%b%", "*", [True, True, False, False, False, False]),
        ("___", "", [True, True, True, False, False, False]),
        ("__/%", "/", [False, False, True, False, False, False]),
        ("55/____", "/", [False, False, False, True, False, False]),
        ("%:%%", ":", [False, False, True, False, False, False]),
        ("55*_100", "*", [False, False, False, True, False, False]),
        ("abc", "abc", [True, False, False, False, False, False]),
    ],
)
def test_string_like(pat, esc, expect):
    expectation = does_not_raise()
    if len(esc) > 1:
        expectation = pytest.raises(ValueError)

    with expectation:
        gs = cudf.Series(["abc", "bab", "99%", "55_100", "", "556100"])
        got = gs.str.like(pat, esc)
        expect = cudf.Series(expect)
        assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "repeats",
    [
        2,
        0,
        -3,
        [5, 4, 3, 2, 6],
        [5, None, 3, 2, 6],
        [0, 0, 0, 0, 0],
        [-1, -2, -3, -4, -5],
        [None, None, None, None, None],
    ],
)
def test_string_repeat(data, repeats):
    ps = pd.Series(["hello", "world", None, "", "!"])
    gs = cudf.from_pandas(ps)

    expect = ps.str.repeat(repeats)
    got = gs.str.repeat(repeats)

    assert_eq(expect, got)


def test_string_cat_str_error():
    gs = cudf.Series(["a", "v", "s"])
    # https://github.com/pandas-dev/pandas/issues/28277
    # ability to pass StringMethods is being removed in future.
    with pytest.raises(
        TypeError,
        match=re.escape(
            "others must be Series, Index, DataFrame, np.ndarrary "
            "or list-like (either containing only strings or "
            "containing only objects of type Series/Index/"
            "np.ndarray[1-dim])"
        ),
    ):
        gs.str.cat(gs.str)


@pytest.mark.parametrize("sep", ["", " ", ",", "|||"])
def test_string_join(ps_gs, sep):
    ps, gs = ps_gs

    expect = ps.str.join(sep)
    got = gs.str.join(sep)

    assert_eq(expect, got)


@pytest.mark.parametrize("pat", [r"(a)", r"(f)", r"([a-z])", r"([A-Z])"])
@pytest.mark.parametrize("expand", [True, False])
@pytest.mark.parametrize(
    "flags,flags_raise", [(0, 0), (re.M | re.S, 0), (re.I, 1)]
)
def test_string_extract(ps_gs, pat, expand, flags, flags_raise):
    ps, gs = ps_gs
    expectation = raise_builder([flags_raise], NotImplementedError)

    with expectation:
        expect = ps.str.extract(pat, flags=flags, expand=expand)
        got = gs.str.extract(pat, flags=flags, expand=expand)

        assert_eq(expect, got)


def test_string_invalid_regex():
    gs = cudf.Series(["a"])
    with pytest.raises(RuntimeError):
        gs.str.extract(r"{\}")


def _cat_convert_seq_to_cudf(others):
    pd_others = others
    if isinstance(pd_others, (pd.Series, pd.Index)):
        gd_others = cudf.from_pandas(pd_others)
    else:
        gd_others = pd_others
    if isinstance(gd_others, (list, tuple)):
        temp_tuple = [
            cudf.from_pandas(elem)
            if isinstance(elem, (pd.Series, pd.Index))
            else elem
            for elem in gd_others
        ]

        if isinstance(gd_others, tuple):
            gd_others = tuple(temp_tuple)
        else:
            gd_others = list(temp_tuple)
    return gd_others


@pytest.mark.parametrize(
    "data",
    [["a", None, "c", None, "e"], ["a", "b", "c", "d", "a"]],
)
@pytest.mark.parametrize(
    "others",
    [
        None,
        ["f", "g", "h", "i", "j"],
        pd.Series(["AbC", "de", "FGHI", "j", "kLm"]),
        pd.Index(["f", "g", "h", "i", "j"]),
        pd.Index(["AbC", "de", "FGHI", "j", "kLm"]),
        [
            np.array(["f", "g", "h", "i", "j"]),
            np.array(["f", "g", "h", "i", "j"]),
        ],
        [
            pd.Series(["f", "g", "h", "i", "j"]),
            pd.Series(["f", "g", "h", "i", "j"]),
        ],
        pytest.param(
            [
                pd.Series(["f", "g", "h", "i", "j"]),
                np.array(["f", "g", "h", "i", "j"]),
            ],
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/5862"
            ),
        ),
        pytest.param(
            (
                pd.Series(["f", "g", "h", "i", "j"]),
                np.array(["f", "a", "b", "f", "a"]),
                pd.Series(["f", "g", "h", "i", "j"]),
                np.array(["f", "a", "b", "f", "a"]),
                np.array(["f", "a", "b", "f", "a"]),
                pd.Index(["1", "2", "3", "4", "5"]),
                np.array(["f", "a", "b", "f", "a"]),
                pd.Index(["f", "g", "h", "i", "j"]),
            ),
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/33436"
            ),
        ),
        [
            pd.Series(
                ["hello", "world", "abc", "xyz", "pqr"],
                index=["a", "b", "c", "d", "e"],
            ),
            pd.Series(
                ["abc", "xyz", "hello", "pqr", "world"],
                index=["a", "b", "c", "d", "e"],
            ),
        ],
        [
            pd.Series(
                ["hello", "world", "abc", "xyz", "pqr"],
                index=[10, 11, 12, 13, 14],
            ),
            pd.Series(
                ["abc", "xyz", "hello", "pqr", "world"],
                index=[10, 15, 11, 13, 14],
            ),
        ],
        [
            pd.Series(
                ["hello", "world", "abc", "xyz", "pqr"],
                index=["1", "2", "3", "4", "5"],
            ),
            pd.Series(
                ["abc", "xyz", "hello", "pqr", "world"],
                index=["1", "2", "3", "4", "5"],
            ),
        ],
    ],
)
@pytest.mark.parametrize("sep", [None, "", " ", ",", "|||"])
@pytest.mark.parametrize("na_rep", [None, "", "null", "a"])
@pytest.mark.parametrize("name", [None, "This is the name"])
def test_string_index_duplicate_str_cat(data, others, sep, na_rep, name):
    pi, gi = pd.Index(data, name=name), cudf.Index(data, name=name)

    pd_others = others
    gd_others = _cat_convert_seq_to_cudf(others)

    got = gi.str.cat(others=gd_others, sep=sep, na_rep=na_rep)
    expect = pi.str.cat(others=pd_others, sep=sep, na_rep=na_rep)

    # TODO: Remove got.sort_values call once we have `join` param support
    # in `.str.cat`
    # https://github.com/rapidsai/cudf/issues/5862

    assert_eq(
        expect.sort_values() if not isinstance(expect, str) else expect,
        got.sort_values() if not isinstance(got, str) else got,
        exact=False,
    )


@pytest.mark.parametrize(
    "others",
    [
        None,
        ["f", "g", "h", "i", "j"],
        ("f", "g", "h", "i", "j"),
        pd.Series(["f", "g", "h", "i", "j"]),
        pd.Series(["AbC", "de", "FGHI", "j", "kLm"]),
        pd.Index(["f", "g", "h", "i", "j"]),
        pd.Index(["AbC", "de", "FGHI", "j", "kLm"]),
        (
            np.array(["f", "g", "h", "i", "j"]),
            np.array(["f", "g", "h", "i", "j"]),
        ),
        [
            np.array(["f", "g", "h", "i", "j"]),
            np.array(["f", "g", "h", "i", "j"]),
        ],
        [
            pd.Series(["f", "g", "h", "i", "j"]),
            pd.Series(["f", "g", "h", "i", "j"]),
        ],
        (
            pd.Series(["f", "g", "h", "i", "j"]),
            pd.Series(["f", "g", "h", "i", "j"]),
        ),
        [
            pd.Series(["f", "g", "h", "i", "j"]),
            np.array(["f", "g", "h", "i", "j"]),
        ],
        (
            pd.Series(["f", "g", "h", "i", "j"]),
            np.array(["f", "g", "h", "i", "j"]),
        ),
        (
            pd.Series(["f", "g", "h", "i", "j"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Series(["f", "g", "h", "i", "j"]),
            np.array(["f", "a", "b", "f", "a"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Index(["1", "2", "3", "4", "5"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Index(["f", "g", "h", "i", "j"]),
        ),
        [
            pd.Index(["f", "g", "h", "i", "j"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Series(["f", "g", "h", "i", "j"]),
            np.array(["f", "a", "b", "f", "a"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Index(["f", "g", "h", "i", "j"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Index(["f", "g", "h", "i", "j"]),
        ],
        [
            pd.Series(["hello", "world", "abc", "xyz", "pqr"]),
            pd.Series(["abc", "xyz", "hello", "pqr", "world"]),
        ],
        [
            pd.Series(
                ["hello", "world", "abc", "xyz", "pqr"],
                index=[10, 11, 12, 13, 14],
            ),
            pd.Series(
                ["abc", "xyz", "hello", "pqr", "world"],
                index=[10, 15, 11, 13, 14],
            ),
        ],
        [
            pd.Series(
                ["hello", "world", "abc", "xyz", "pqr"],
                index=["10", "11", "12", "13", "14"],
            ),
            pd.Series(
                ["abc", "xyz", "hello", "pqr", "world"],
                index=["10", "11", "12", "13", "14"],
            ),
        ],
        [
            pd.Series(
                ["hello", "world", "abc", "xyz", "pqr"],
                index=["10", "11", "12", "13", "14"],
            ),
            pd.Series(
                ["abc", "xyz", "hello", "pqr", "world"],
                index=["10", "15", "11", "13", "14"],
            ),
        ],
        [
            pd.Series(
                ["hello", "world", "abc", "xyz", "pqr"],
                index=["1", "2", "3", "4", "5"],
            ),
            pd.Series(
                ["abc", "xyz", "hello", "pqr", "world"],
                index=["10", "11", "12", "13", "14"],
            ),
        ],
    ],
)
@pytest.mark.parametrize("sep", [None, "", " ", "|", ",", "|||"])
@pytest.mark.parametrize("na_rep", [None, "", "null", "a"])
@pytest.mark.parametrize(
    "index",
    [["1", "2", "3", "4", "5"]],
)
def test_string_cat(ps_gs, others, sep, na_rep, index):
    ps, gs = ps_gs

    pd_others = others
    gd_others = _cat_convert_seq_to_cudf(others)

    expect = ps.str.cat(others=pd_others, sep=sep, na_rep=na_rep)
    got = gs.str.cat(others=gd_others, sep=sep, na_rep=na_rep)
    assert_eq(expect, got)

    ps.index = index
    gs.index = index

    expect = ps.str.cat(others=ps.index, sep=sep, na_rep=na_rep)
    got = gs.str.cat(others=gs.index, sep=sep, na_rep=na_rep)

    assert_eq(expect, got)

    expect = ps.str.cat(others=[ps.index, ps.index], sep=sep, na_rep=na_rep)
    got = gs.str.cat(others=[gs.index, gs.index], sep=sep, na_rep=na_rep)

    assert_eq(expect, got)

    expect = ps.str.cat(others=(ps.index, ps.index), sep=sep, na_rep=na_rep)
    got = gs.str.cat(others=(gs.index, gs.index), sep=sep, na_rep=na_rep)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        ["1", "2", "3", "4", "5"],
        ["a", "b", "c", "d", "e"],
        ["a", "b", "c", None, "e"],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        None,
        ["f", "g", "h", "i", "j"],
        ("f", "g", "h", "i", "j"),
        pd.Series(["f", "g", "h", "i", "j"]),
        pd.Series(["AbC", "de", "FGHI", "j", "kLm"]),
        pd.Index(["f", "g", "h", "i", "j"]),
        pd.Index(["AbC", "de", "FGHI", "j", "kLm"]),
        (
            np.array(["f", "g", "h", "i", "j"]),
            np.array(["f", "g", "h", "i", "j"]),
        ),
        [
            np.array(["f", "g", "h", "i", "j"]),
            np.array(["f", "g", "h", "i", "j"]),
        ],
        [
            pd.Series(["f", "g", "h", "i", "j"]),
            pd.Series(["f", "g", "h", "i", "j"]),
        ],
        (
            pd.Series(["f", "g", "h", "i", "j"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Series(["f", "g", "h", "i", "j"]),
            np.array(["f", "a", "b", "f", "a"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Index(["1", "2", "3", "4", "5"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Index(["f", "g", "h", "i", "j"]),
        ),
        [
            pd.Index(["f", "g", "h", "i", "j"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Series(["f", "g", "h", "i", "j"]),
            np.array(["f", "a", "b", "f", "a"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Index(["f", "g", "h", "i", "j"]),
            np.array(["f", "a", "b", "f", "a"]),
            pd.Index(["f", "g", "h", "i", "j"]),
        ],
        [
            pd.Series(
                ["hello", "world", "abc", "xyz", "pqr"],
                index=["a", "b", "c", "d", "e"],
            ),
            pd.Series(
                ["abc", "xyz", "hello", "pqr", "world"],
                index=["a", "b", "c", "d", "e"],
            ),
        ],
        [
            pd.Series(
                ["hello", "world", "abc", "xyz", "pqr"],
                index=[10, 11, 12, 13, 14],
            ),
            pd.Series(
                ["abc", "xyz", "hello", "pqr", "world"],
                index=[10, 15, 11, 13, 14],
            ),
        ],
        [
            pd.Series(
                ["hello", "world", "abc", "xyz", "pqr"],
                index=["1", "2", "3", "4", "5"],
            ),
            pd.Series(
                ["abc", "xyz", "hello", "pqr", "world"],
                index=["1", "2", "3", "4", "5"],
            ),
        ],
    ],
)
@pytest.mark.parametrize("sep", [None, "", " ", "|", "|||"])
@pytest.mark.parametrize("na_rep", [None, "", "null", "a"])
@pytest.mark.parametrize("name", [None, "This is the name"])
def test_string_index_str_cat(data, others, sep, na_rep, name):
    pi, gi = pd.Index(data, name=name), cudf.Index(data, name=name)

    pd_others = others
    gd_others = _cat_convert_seq_to_cudf(others)

    expect = pi.str.cat(others=pd_others, sep=sep, na_rep=na_rep)
    got = gi.str.cat(others=gd_others, sep=sep, na_rep=na_rep)

    assert_eq(
        expect,
        got,
        exact=False,
    )


def test_string_len(ps_gs):
    ps, gs = ps_gs

    expect = ps.str.len()
    got = gs.str.len()

    # Can't handle nulls in Pandas so use PyArrow instead
    # Pandas will return as a float64 so need to typecast to int32
    expect = pa.array(expect, from_pandas=True).cast(pa.int32())
    got = got.to_arrow()
    assert pa.Array.equals(expect, got)


def test_string_concat():
    data1 = ["a", "b", "c", "d", "e"]
    data2 = ["f", "g", "h", "i", "j"]
    index = [1, 2, 3, 4, 5]

    ps1 = pd.Series(data1, index=index)
    ps2 = pd.Series(data2, index=index)
    gs1 = cudf.Series(data1, index=index)
    gs2 = cudf.Series(data2, index=index)

    expect = pd.concat([ps1, ps2])
    got = concat([gs1, gs2])

    assert_eq(expect, got)

    expect = ps1.str.cat(ps2)
    got = gs1.str.cat(gs2)

    assert_eq(expect, got)


@pytest.mark.parametrize("name", [None, "new name", 123])
def test_string_misc_name(ps_gs, name):
    ps, gs = ps_gs
    ps.name = name
    gs.name = name

    expect = ps.str.slice(0, 1)
    got = gs.str.slice(0, 1)

    assert_eq(expect, got)
    assert_eq(ps + ps, gs + gs)
    assert_eq(ps + "RAPIDS", gs + "RAPIDS")
    assert_eq("RAPIDS" + ps, "RAPIDS" + gs)
