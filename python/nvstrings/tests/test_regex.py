# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import pytest
import numpy as np
import pandas as pd
import nvstrings

from utils import assert_eq


@pytest.mark.parametrize('pattern', ['\\d',
                                     '\\w+',
                                     '\\s',
                                     '\\S',
                                     '^.*\\\\.*$',
                                     '[1-5]+',
                                     '[a-h]+',
                                     '[A-H]+',
                                     '\n',
                                     'b.\\s*\n',
                                     '.*c',
                                     '\\d\\d:\\d\\d:\\d\\d',
                                     '\\d\\d?:\\d\\d?:\\d\\d?',
                                     '[Hh]ello [Ww]orld',
                                     '\\bworld\\b'
                                     ])
def test_contains(pattern):
    s = [
        '5',
        'hej',
        '\t \n',
        '12345',
        '\\',
        'd',
        'c:\\Tools',
        '+27',
        '1c2',
        '1C2',
        '0:00:0',
        '0:0:00',
        '00:0:0',
        '00:00:0',
        '00:0:00',
        '0:00:00',
        '00:00:00',
        'Hello world !',
        'Hello world!   ',
        'Hello worldcup  !',
        '0123456789',
        '1C2',
        'Xaa',
        'abcdefghxxx',
        'ABCDEFGH',
        'abcdefgh',
        'abc def',
        'abc\ndef',
        'aa\r\nbb\r\ncc\r\n\r\n',
        'abcabc'
    ]
    pstrs = pd.Series(s)
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.contains(pattern)
    expected = pstrs.str.contains(pattern).values
    assert_eq(got, expected)


@pytest.mark.parametrize('find', ["@\\S+", "(?:@|https?://)\\S+"])
@pytest.mark.parametrize('replace', ["***", ""])
def test_replace(find, replace):
    s = ["hello @abc @def world", "The quick brown @fox jumps", "over the",
         "lazy @dog", "hello http://www.world.com I'm here @home"]
    pstrs = pd.Series(s)
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.replace(find, replace)
    expected = pstrs.str.replace(find, replace).values
    assert_eq(got, expected)


def test_replace_multi():
    s = ["xxx 1281151 xxxxxx xxxxxxx xxxx xxxx - xxxxx xxxx xx 24",
         "2-xxxx xxxxxxxxxxx xxxxxxxxxx xxx26x4xxx xxxxxxxxxxxx xxxxx xxxxx"]
    pstrs = pd.Series(s)
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.replace(r'\b\d+\b', '*****')
    expected = pstrs.str.replace(r'\b\d+\b', '*****').values
    assert_eq(got, expected)


@pytest.mark.parametrize('pattern', ['[hH]',
                                     '[bB][aA]',
                                     ])
def test_match(pattern):
    s = ["hello", "and héllo", None, ""]
    pstrs = pd.Series(s)
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.match(pattern)
    expected = pstrs.str.match(pattern).values
    assert_eq(got, expected)


@pytest.mark.parametrize('pattern', ['a',
                                     '[aA]',
                                     ])
def test_count(pattern):
    s = ["hello", "and héllo", 'this was empty', ""]
    pstrs = pd.Series(s)
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.count(pattern)
    expected = pstrs.str.count(pattern).values
    assert_eq(got, expected)


def test_findall():
    pattern = '[aA]'
    s = ["hello", "and héllo", 'this was empty', ""]
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.findall(pattern)[0]
    expected = [None, 'a', 'a', None]
    assert_eq(got, expected)


def test_findall_record():
    pattern = '[aA]'
    s = ["hello", "and héllo", 'this was empty', "", 'another']
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.findall_record(pattern)
    expected = [[], ['a'], ['a'], [], ['a']]
    for i in range(len(got)):
        assert got[i].to_host() == expected[i]


def test_extract():
    pattern = r'Flight:([A-Z]+)(\d+)'
    s = ['ALA-PEK Flight:HU7934', 'HKT-PEK Flight:CA822',
         'FRA-PEK Flight:LA8769', 'FRA-PEK Flight:LH7332', '', None,
         'Flight:ZZ']
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.extract(pattern)
    expected = np.array([['HU', '7934'],
                         ['CA', '822'],
                         ['LA', '8769'],
                         ['LH', '7332'],
                         [None, None],
                         [None, None],
                         [None, None]])
    assert_eq(got[0], expected[:, 0])
    assert_eq(got[1], expected[:, 1])


def test_extract_record():
    pattern = r'Flight:([A-Z]+)(\d+)'
    s = ['ALA-PEK Flight:HU7934', 'HKT-PEK Flight:CA822',
         'FRA-PEK Flight:LA8769', 'FRA-PEK Flight:LH7332', '', None,
         'Flight:ZZ']
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.extract_record(pattern)
    expected = np.array([['HU', '7934'],
                         ['CA', '822'],
                         ['LA', '8769'],
                         ['LH', '7332'],
                         [None, None],
                         [None, None],
                         [None, None]])

    for i in range(len(got)):
        assert_eq(got[i], expected[i, :])


@pytest.mark.parametrize('find', ['(\\d)(\\d)',
                                  '(\\d)(\\d)',
                                  '(\\d)(\\d)',
                                  '(\\d)(\\d)',
                                  "([a-z])-([a-z])",
                                  "([a-z])-([a-zé])",
                                  "([a-z])-([a-z])",
                                  "([a-z])-([a-zé])"
                                  ])
@pytest.mark.parametrize('replace', [
    '\\1-\\2',
    'V\\2-\\1',
    pytest.param('V\\1-\\3', marks=[pytest.mark.xfail(
         reason='Pandas fails with this backreference group 3')]),
    pytest.param('V\\3-\\2', marks=[pytest.mark.xfail(
         reason='Pandas fails with this backreference group 3')]),
    "\\1 \\2",
    "\\2 \\1",
    "X\\1+\\2Z",
    "X\\1+\\2Z"
])
def test_replace_with_backrefs(find, replace):
    s = ["A543", "Z756", "", None, 'tést-string', 'two-thréé four-fivé',
         'abcd-éfgh', 'tést-string-again']
    pstrs = pd.Series(s)
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.replace_with_backrefs(find, replace)
    expected = pstrs.str.replace(find, replace).values
    assert_eq(got, expected)


@pytest.mark.parametrize('pattern', [
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home",
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz"
])
def test_contains_large_regex(pattern):
    s = ["hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home", "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890", "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"]
    pstrs = pd.Series(s)
    strs = nvstrings.to_device(s)
    got = strs.contains(pattern)
    expected = pstrs.str.contains(pattern)
    assert_eq(got, expected)
