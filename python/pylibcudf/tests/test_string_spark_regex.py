# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Regex patterns derived from spark-rapids integration tests:
# https://github.com/NVIDIA/spark-rapids/blob/main/integration_tests/src/main/python/regexp_test.py
#
# Tests both Thompson NFA (DEFAULT) and Glushkov NFA (GLUSHKOV) engines.
# Patterns using assertions (^, $, \b, \B) or >64 positions automatically
# fall back to Thompson inside cuDF, so all patterns are valid for both flags.

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spark_strings():
    """Rich string array that exercises the spark-rapids regex patterns."""
    return pa.array(
        [
            "abc",
            "aabbc",
            "123abc",
            "abc123def",
            "a1b2c3",
            "boo:and:foo",
            "foo:boo:",
            "TEST",
            "TESTaaa",
            "TEST123",
            "abcd",
            "aaa",
            "bbb",
            "ccc",
            "abb",
            "ab",
            "aab",
            "a\nb",
            "a\tb",
            "a b",
            "",
            "aaabbb",
            "abcabc",
            "aa|bb",
            "foobar",
            "12345",
            "abcdef",
            "ABCDEF",
            "AbCdEf",
            None,
        ]
    )


@pytest.fixture(
    params=[
        pytest.param(
            plc.strings.regex_flags.RegexFlags.DEFAULT,
            id="DEFAULT",
        ),
        pytest.param(
            plc.strings.regex_flags.RegexFlags.GLUSHKOV,
            id="GLUSHKOV",
        ),
    ],
    scope="module",
)
def regex_flags(request):
    return request.param


def _make_prog(pattern, flags):
    return plc.strings.regex_program.RegexProgram.create(pattern, flags)


# ---------------------------------------------------------------------------
# contains_re tests — patterns from spark-rapids rlike / regexp / regexp_like
# ---------------------------------------------------------------------------


# Basic quantifiers (test_rlike, test_regexp, test_regexp_like)
@pytest.mark.parametrize(
    "pattern",
    [
        "a{2}",
        "a{1,3}",
        "a{1,}",
        "a[bc]d",
    ],
)
def test_contains_re_basic(spark_strings, regex_flags, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# Alternation patterns (test_regexp_choice, test_rlike_rewrite_optimization)
@pytest.mark.parametrize(
    "pattern",
    [
        "aaa|bbb|ccc",
        "1|2|3|4|5|6",
        "[abcd]|[123]",
        "aaa|bbb",
        "aaa|(bbb|ccc)",
        ".*.*(aaa|bbb).*.*",
        "^.*(aaa|bbb|ccc)",
        "abd1a$|^ab2a",
        "[ab]+|^cd1",
    ],
)
def test_contains_re_alternation(spark_strings, regex_flags, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# Anchor + wildcard patterns (test_rlike_rewrite_optimization)
@pytest.mark.parametrize(
    "pattern",
    [
        "^abb",
        "^.*(aaa)",
        "^(abb)(.*)",
        "abb(.*)",
        "(.*)(abb)(.*)",
        "ab(.*)cd",
        "(.*)(.*)abb",
    ],
)
def test_contains_re_anchors_wildcards(spark_strings, regex_flags, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# Bounded repetition (test_rlike_rewrite_optimization, test_regexp)
@pytest.mark.parametrize(
    "pattern",
    [
        "ab[a-c]{3}",
        "a[a-c]{1,3}",
        "a[a-c]{1,}",
        "a[a-c]+",
        "(ab)([a-c]{1})",
        "(ab[a-c]{1})",
        "a{6}",
        "a{1,6}",
    ],
)
def test_contains_re_bounded_repetition(spark_strings, regex_flags, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# Non-capturing groups / complex quantifiers (test_regexp_memory_ok)
@pytest.mark.parametrize(
    "pattern",
    [
        "(?:12345)+",
        "(?:aa)+",
        "abcdef",
        "(1)(2)(3)",
    ],
)
def test_contains_re_groups(spark_strings, regex_flags, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# Predefined character classes (test_character_classes, test_regexp_whitespace,
# test_regexp_replace_digit, test_regexp_replace_word)
@pytest.mark.parametrize(
    "pattern",
    [
        r"\d",
        r"\D",
        r"[0-9]",
        r"[^0-9]",
        r"\w",
        r"[a-zA-Z_0-9]",
        r"\s",
        r"\S",
        r"[abcd]+\s+[0-9]+",
        r"\S{3}",
        r"[^\n\r]",
    ],
)
def test_contains_re_char_classes(spark_strings, regex_flags, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# \W: cuDF uses ASCII semantics — only [^a-zA-Z0-9_] are non-word chars,
# unlike pyarrow/RE2 which is Unicode-aware. Test with hardcoded expectations.
def test_contains_re_nonword(regex_flags):
    arr = pa.array(
        ["abc", "abc123", "abc!", "a b", "aa|bb", "12345", "", None]
    )
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(arr),
        _make_prog(r"\W", regex_flags),
    )
    # cuDF \W matches non-ASCII-word characters: space, !, |
    expect = pa.array([False, False, True, True, True, False, False, None])
    assert_column_eq(expect, got)


# Escape / character class edge cases (test_rlike_escape, test_rlike_missing_escape)
@pytest.mark.parametrize(
    "pattern",
    [
        r"a[\-]",
        r"a[+-]",
        r"a[a-b-]",
        r"[a-z]{3,}",
        r"^[A-Z]{2,}",
    ],
)
def test_contains_re_escape_edge_cases(spark_strings, regex_flags, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# ---------------------------------------------------------------------------
# replace_re tests — patterns from spark-rapids regexp_replace
# ---------------------------------------------------------------------------

REPLACE_REPL = "X"  # replacement string used in all replace tests


# Basic replace patterns (test_re_replace, test_regexp_replace)
@pytest.mark.parametrize(
    "pattern",
    [
        "TEST",
        "[A-Z]+",
        "a",
        "[^xyz]",
        "a|b|c",
    ],
)
def test_replace_re_basic(spark_strings, regex_flags, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)


# Repetition quantifiers (test_re_replace_repetition)
@pytest.mark.parametrize(
    "pattern",
    [
        "[E]+",
        "[A]+",
        "[A-Z]+",
    ],
)
def test_replace_re_quantifiers(spark_strings, regex_flags, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)


# Negated character classes (test_regexp_replace_character_set_negated)
@pytest.mark.parametrize(
    "pattern",
    [
        "[^a]",
        r"[^a\r\n]",
        r"[^\r\n]",
        r"[^\r]",
        r"[^\n]",
    ],
)
def test_replace_re_negated_classes(spark_strings, regex_flags, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)


# Digit and word classes (test_regexp_replace_digit, test_regexp_replace_word)
# \D and \W are excluded from pyarrow comparison: cuDF uses ASCII semantics for
# \w/\W (only [a-zA-Z0-9_]), while pyarrow/RE2 is Unicode-aware.
@pytest.mark.parametrize(
    "pattern",
    [
        r"\d",
        r"[0-9]",
        r"[^0-9]",
        r"\w",
        r"[a-zA-Z_0-9]",
        r"[^a-zA-Z_0-9]",
    ],
)
def test_replace_re_digit_word(spark_strings, regex_flags, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)


# \D replace: cuDF ASCII semantics — non-digits include letters, spaces, punctuation
def test_replace_re_nondigit(regex_flags):
    arr = pa.array(["abc", "a1b2", "123", "a b", "", None])
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(arr),
        _make_prog(r"\D", regex_flags),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pa.array(["XXX", "X1X2", "123", "XXX", "", None])
    assert_column_eq(expect, got)


# \W replace: cuDF ASCII semantics — non-word chars are non-[a-zA-Z0-9_]
def test_replace_re_nonword(regex_flags):
    arr = pa.array(["abc", "a b", "a!b", "aa|bb", "123", "", None])
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(arr),
        _make_prog(r"\W", regex_flags),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pa.array(["abc", "aXb", "aXb", "aaXbb", "123", "", None])
    assert_column_eq(expect, got)


# Multi-alternation (test_regexp_replace_multi_optimization)
@pytest.mark.parametrize(
    "pattern",
    [
        "aa|bb",
        "aa|bb|cc",
        "aa|bb|cc|dd",
        "aa|bb|cc|dd|ee",
        "aa|bb|cc|dd|ee|ff",
        "(aa)|(bb)",
        "(aa)|(bb)|(cc)",
        "(aa|bb)|(cc|dd)",
    ],
)
def test_replace_re_multi_alternation(spark_strings, regex_flags, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)


# Non-capturing groups (test_regexp_replace)
@pytest.mark.parametrize(
    "pattern",
    [
        "(?:aa)+",
        "([^x])|([^y])",
    ],
)
def test_replace_re_noncapturing(spark_strings, regex_flags, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern, regex_flags),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)
