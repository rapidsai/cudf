# Copyright (c) 2019, NVIDIA CORPORATION.

import pytest
from pandas.util.testing import assert_series_equal

import cudf
from cudf.tests.utils import assert_eq


def test_tokenize():
    strings = cudf.Series(
        [
            "the quick fox jumped over the lazy dog",
            "the siamésé cat jumped under the sofa",
            None,
            "",
        ]
    )

    expected = cudf.Series(
        [
            "the",
            "quick",
            "fox",
            "jumped",
            "over",
            "the",
            "lazy",
            "dog",
            "the",
            "siamésé",
            "cat",
            "jumped",
            "under",
            "the",
            "sofa",
        ]
    )

    actual = strings.str.tokenize()

    assert type(expected) == type(actual)
    assert_series_equal(expected.to_pandas(), actual.to_pandas())


@pytest.mark.parametrize(
    "delimiter, expected_token_counts",
    [
        ("", [10, 9, 0, 0, 5]),
        ("o", [6, 3, 0, 0, 1]),
        (["a", "e", "i", "o", "u"], [13, 13, 0, 0, 6]),
        (["a", "e", "i", "o"], [12, 11, 0, 0, 6]),
    ],
)
def test_token_count(delimiter, expected_token_counts):
    strings = cudf.Series(
        [
            "the quick brown fox jumped over the lazy brown dog",
            "the sable siamésé cat jumped under the brown sofa",
            None,
            "",
            "test_str\x01test_str\x02test_str\x03test_str\x04test_str\x05",
        ]
    )

    expected = cudf.Series(expected_token_counts)

    actual = strings.str.token_count(delimiter)

    assert type(expected) == type(actual)
    assert_series_equal(
        expected.to_pandas(), actual.to_pandas(), check_dtype=False
    )


def test_normalize_spaces():
    strings = cudf.Series(
        [
            " the\t quick fox  jumped over the lazy dog",
            "the siamésé cat\f jumped\t\tunder the sofa  ",
            None,
            "",
        ]
    )
    expected = cudf.Series(
        [
            "the quick fox jumped over the lazy dog",
            "the siamésé cat jumped under the sofa",
            None,
            "",
        ]
    )

    actual = strings.str.normalize_spaces()

    assert type(expected) == type(actual)
    assert_series_equal(expected.to_pandas(), actual.to_pandas())


@pytest.mark.parametrize(
    "n, separator, expected_values",
    [
        (
            2,
            "_",
            [
                "this_is",
                "is_my",
                "my_favorite",
                "favorite_book",
                "book_on",
                "on_my",
                "my_bookshelf",
            ],
        ),
        (
            3,
            "-",
            [
                "this-is-my",
                "is-my-favorite",
                "my-favorite-book",
                "favorite-book-on",
                "book-on-my",
                "on-my-bookshelf",
            ],
        ),
    ],
)
def test_ngrams(n, separator, expected_values):
    strings = cudf.Series(
        ["this", "is", "my", "favorite", "book", "on", "my", "bookshelf"]
    )

    expected = cudf.Series(expected_values)

    actual = strings.str.ngrams(n=n, separator=separator)

    assert type(expected) == type(actual)
    assert_series_equal(expected.to_pandas(), actual.to_pandas())


@pytest.mark.parametrize(
    "n, separator, expected_values",
    [
        (
            2,
            "_",
            [
                "this_is",
                "is_my",
                "my_favorite",
                "book_on",
                "on_my",
                "my_bookshelf",
            ],
        ),
        (
            3,
            "-",
            ["this-is-my", "is-my-favorite", "book-on-my", "on-my-bookshelf"],
        ),
    ],
)
def test_ngrams_tokenize(n, separator, expected_values):
    strings = cudf.Series(["this is my favorite", "book on my bookshelf"])

    expected = cudf.Series(expected_values)

    actual = strings.str.ngrams_tokenize(n=n, separator=separator)

    assert type(expected) == type(actual)
    assert_series_equal(expected.to_pandas(), actual.to_pandas())


def test_character_tokenize_series():
    sr = cudf.Series(
        [
            "hello world",
            "sdf",
            None,
            "goodbye, one-two:three~four+five_six@sev"
            "en#eight^nine heŒŽ‘•™œ$µ¾ŤƠé Ǆ",
        ]
    )
    expected = cudf.Series(
        [
            "h",
            "e",
            "l",
            "l",
            "o",
            " ",
            "w",
            "o",
            "r",
            "l",
            "d",
            "s",
            "d",
            "f",
            "g",
            "o",
            "o",
            "d",
            "b",
            "y",
            "e",
            ",",
            " ",
            "o",
            "n",
            "e",
            "-",
            "t",
            "w",
            "o",
            ":",
            "t",
            "h",
            "r",
            "e",
            "e",
            "~",
            "f",
            "o",
            "u",
            "r",
            "+",
            "f",
            "i",
            "v",
            "e",
            "_",
            "s",
            "i",
            "x",
            "@",
            "s",
            "e",
            "v",
            "e",
            "n",
            "#",
            "e",
            "i",
            "g",
            "h",
            "t",
            "^",
            "n",
            "i",
            "n",
            "e",
            " ",
            "h",
            "e",
            "Œ",
            "Ž",
            "‘",
            "•",
            "™",
            "œ",
            "$",
            "µ",
            "¾",
            "Ť",
            "Ơ",
            "é",
            " ",
            "Ǆ",
        ]
    )

    actual = sr.str.character_tokenize()
    assert_eq(expected, actual)

    sr = cudf.Series([""])
    expected = cudf.Series([], dtype="object")

    actual = sr.str.character_tokenize()
    assert_eq(expected, actual)

    sr = cudf.Series(["a"])
    expected = cudf.Series(["a"])

    actual = sr.str.character_tokenize()
    assert_eq(expected, actual)


def test_character_tokenize_index():
    sr = cudf.core.index.as_index(
        [
            "hello world",
            "sdf",
            None,
            "goodbye, one-two:three~four+five_six@sev"
            "en#eight^nine heŒŽ‘•™œ$µ¾ŤƠé Ǆ",
        ]
    )
    expected = cudf.core.index.as_index(
        [
            "h",
            "e",
            "l",
            "l",
            "o",
            " ",
            "w",
            "o",
            "r",
            "l",
            "d",
            "s",
            "d",
            "f",
            "g",
            "o",
            "o",
            "d",
            "b",
            "y",
            "e",
            ",",
            " ",
            "o",
            "n",
            "e",
            "-",
            "t",
            "w",
            "o",
            ":",
            "t",
            "h",
            "r",
            "e",
            "e",
            "~",
            "f",
            "o",
            "u",
            "r",
            "+",
            "f",
            "i",
            "v",
            "e",
            "_",
            "s",
            "i",
            "x",
            "@",
            "s",
            "e",
            "v",
            "e",
            "n",
            "#",
            "e",
            "i",
            "g",
            "h",
            "t",
            "^",
            "n",
            "i",
            "n",
            "e",
            " ",
            "h",
            "e",
            "Œ",
            "Ž",
            "‘",
            "•",
            "™",
            "œ",
            "$",
            "µ",
            "¾",
            "Ť",
            "Ơ",
            "é",
            " ",
            "Ǆ",
        ]
    )

    actual = sr.str.character_tokenize()
    assert_eq(expected, actual)

    sr = cudf.core.index.as_index([""])
    expected = cudf.core.index.StringIndex([], dtype="object")

    actual = sr.str.character_tokenize()
    assert_eq(expected, actual)

    sr = cudf.core.index.as_index(["a"])
    expected = cudf.core.index.as_index(["a"])

    actual = sr.str.character_tokenize()
    assert_eq(expected, actual)
