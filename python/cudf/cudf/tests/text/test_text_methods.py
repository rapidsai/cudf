# Copyright (c) 2019-2023, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.testing._utils import assert_eq


def test_tokenize():
    strings = cudf.Series(
        [
            "the quick fox jumped over the lazy dog",
            "the siamésé cat jumped under the sofa",
            None,
            "",
        ]
    )

    expected_values = cudf.Series(
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
    expected_index = strings.index.repeat(strings.str.token_count())
    expected = cudf.Series(expected_values, index=expected_index)

    actual = strings.str.tokenize()

    assert type(expected) == type(actual)
    assert_eq(expected, actual)


def test_tokenize_delimiter():
    strings = cudf.Series(
        [
            "the quick fox jumped over the lazy dog",
            "the siamésé cat jumped under the sofa",
            None,
            "",
        ]
    )

    expected_values = cudf.Series(
        [
            "the quick f",
            "x jumped ",
            "ver the lazy d",
            "g",
            "the siamésé cat jumped under the s",
            "fa",
        ]
    )
    expected_index = strings.index.repeat(strings.str.token_count("o"))
    expected = cudf.Series(expected_values, index=expected_index)

    actual = strings.str.tokenize(delimiter="o")

    assert type(expected) == type(actual)
    assert_eq(expected, actual)


def test_detokenize():
    strings = cudf.Series(
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

    indices = cudf.Series([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
    actual = strings.str.detokenize(indices)
    expected = cudf.Series(
        [
            "the quick fox",
            "jumped over",
            "the lazy dog",
            "the siamésé cat jumped under the sofa",
        ]
    )
    assert type(expected) == type(actual)
    assert_eq(expected, actual)

    indices = cudf.Series(
        [4, 0, 0, 0, 0, 4, 1, 1, 4, 2, 2, 2, 2, 4, 3], dtype=np.int8
    )
    actual = strings.str.detokenize(indices, "+")
    expected = cudf.Series(
        [
            "quick+fox+jumped+over",
            "lazy+dog",
            "siamésé+cat+jumped+under",
            "sofa",
            "the+the+the+the",
        ]
    )
    assert type(expected) == type(actual)
    assert_eq(expected, actual)


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
    assert_eq(expected, actual, check_dtype=False)


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
    assert_eq(expected, actual)


def test_normalize_characters():
    strings = cudf.Series(
        ["乾 \t 乿", "ĂĆCĖÑTÜATE", "âscénd, Descend", "", None, "Stock^ $1"]
    )
    expected = cudf.Series(
        [
            " 乾     乿 ",
            "accentuate",
            "ascend ,  descend",
            "",
            None,
            "stock ^   $ 1",
        ]
    )

    actual = strings.str.normalize_characters()
    assert type(expected) == type(actual)
    assert_eq(expected, actual)

    expected = cudf.Series(
        [
            " 乾     乿 ",
            "ĂĆCĖÑTÜATE",
            "âscénd ,  Descend",
            "",
            None,
            "Stock ^   $ 1",
        ]
    )
    actual = strings.str.normalize_characters(do_lower=False)
    assert type(expected) == type(actual)
    assert_eq(expected, actual)


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
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "n, expected_values, expected_index, as_list",
    [
        (
            2,
            [
                "th",
                "hi",
                "is",
                "is",
                "my",
                "bo",
                "oo",
                "ok",
                "he",
                "er",
                "re",
                cudf.NA,
            ],
            [1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 6],
            False,
        ),
        (
            3,
            [
                "thi",
                "his",
                cudf.NA,
                cudf.NA,
                "boo",
                "ook",
                "her",
                "ere",
                cudf.NA,
            ],
            [1, 1, 2, 3, 4, 4, 5, 5, 6],
            False,
        ),
        (
            3,
            [["thi", "his"], [], [], ["boo", "ook"], ["her", "ere"], []],
            [1, 2, 3, 4, 5, 6],
            True,
        ),
    ],
)
def test_character_ngrams(n, expected_values, expected_index, as_list):
    strings = cudf.Series(
        ["this", "is", "my", "book", "here", ""], index=[1, 2, 3, 4, 5, 6]
    )

    expected = cudf.Series(expected_values, index=expected_index)

    actual = strings.str.character_ngrams(n=n, as_list=as_list)

    assert type(expected) == type(actual)
    assert_eq(expected, actual)


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
    assert_eq(expected, actual)


def test_character_tokenize_series():
    sr = cudf.Series(
        [
            "hello world",
            "sdf",
            None,
            (
                "goodbye, one-two:three~four+five_six@sev"
                "en#eight^nine heŒŽ‘•™œ$µ¾ŤƠé Ǆ"
            ),
        ]
    )
    expected_values = cudf.Series(
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
    expected_index = sr.index.repeat(sr.str.len().fillna(0))
    expected = cudf.Series(expected_values, index=expected_index)

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
            (
                "goodbye, one-two:three~four+five_six@sev"
                "en#eight^nine heŒŽ‘•™œ$µ¾ŤƠé Ǆ"
            ),
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

    sr = cudf.Index([""])
    expected = cudf.Index([], dtype="object")

    actual = sr.str.character_tokenize()
    assert_eq(expected, actual)

    sr = cudf.core.index.as_index(["a"])
    expected = cudf.core.index.as_index(["a"])

    actual = sr.str.character_tokenize()
    assert_eq(expected, actual)


def test_text_replace_tokens():
    sr = cudf.Series(["this is me", "theme music", ""])
    targets = cudf.Series(["is", "me"])

    expected = cudf.Series(["this _ _", "theme music", ""])
    actual = sr.str.replace_tokens(targets, "_")

    assert_eq(expected, actual)

    replacements = cudf.Series(["IS", "ME"])
    expected = cudf.Series(["this IS ME", "theme music", ""])
    actual = sr.str.replace_tokens(targets, replacements)

    assert_eq(expected, actual)

    sr = cudf.Series(
        [
            "this is a small text ☕",
            "this \t\t is ; ; - + a looooooooooonnnnnnnggggggg text \n\t",
            "emptyme",
        ],
    )
    targets = cudf.Series(
        ["a", "☕", "\t", "looooooooooonnnnnnnggggggg", "emptyme"]
    )
    replacements = cudf.Series(["the", "🚒", "🚒🚒🚒🚒", "🔥🔥", ""])

    expected = cudf.Series(
        [
            "this is the small text 🚒",
            "this \t\t is ; ; - + the 🔥🔥 text \n\t",
            "",
        ]
    )
    actual = sr.str.replace_tokens(targets, replacements)

    assert_eq(expected, actual)

    sr = cudf.Series(
        ["All-we-need;is;🔥", "\tall-we-need0is;🌊", "all;we:need+is;🌬"]
    )
    targets = cudf.Series(["🌬", "🔥", "🌊"])
    replacements = "🚰"

    expected = cudf.Series(
        ["All-we-need;is;🚰", "\tall-we-need0is;🚰", "all;we:need+is;🚰"]
    )
    actual = sr.str.replace_tokens(targets, replacements, delimiter=";")

    assert_eq(expected, actual)
    assert_eq(sr, sr.str.replace_tokens(targets, replacements))
    assert_eq(sr, sr.str.replace_tokens([""], [""]))


def test_text_replace_tokens_error_cases():
    sr = cudf.Series(["this is me", "theme music", ""])

    with pytest.raises(
        TypeError,
        match="targets should be an array-like or a Series object, "
        "found <class 'str'>",
    ):
        sr.str.replace_tokens("me", ["a"])

    with pytest.raises(
        ValueError,
        match="targets and replacements should be same size"
        " sequences unless replacements is a string.",
    ):
        sr.str.replace_tokens(["a"], ["me", "ki"])

    with pytest.raises(
        TypeError,
        match="replacements should be an str, array-like or Series object,"
        " found <class 'set'>",
    ):
        sr.str.replace_tokens(["a"], {"s"})

    with pytest.raises(
        TypeError,
        match="Type of delimiter should be a string, found <class 'list'>",
    ):
        sr.str.replace_tokens(["a"], ["s"], delimiter=["a", "b"])


def test_text_filter_tokens():
    sr = cudf.Series(["the quick brown fox jumped", "over the lazy dog", ""])

    expected = cudf.Series([" quick brown  jumped", "   ", ""])
    actual = sr.str.filter_tokens(5)
    assert_eq(expected, actual)

    expected = cudf.Series(["🔥 quick brown 🔥 jumped", "🔥 🔥 🔥 🔥", ""])
    actual = sr.str.filter_tokens(5, "🔥")
    assert_eq(expected, actual)

    sr = cudf.Series(
        ["All-we-need;is;🔥", "\tall-we-need0is;🌊", "all;we:need+is;🌬"]
    )
    expected = cudf.Series(
        ["All-we-need;is;--", "\tall-we-need0is;--", "all;we:need+is;--"]
    )
    actual = sr.str.filter_tokens(2, "--", ";")
    assert_eq(expected, actual)

    assert_eq(sr, sr.str.filter_tokens(1))


def test_text_filter_tokens_error_cases():
    sr = cudf.Series(["abc", "def", ""])

    with pytest.raises(
        TypeError,
        match="Type of replacement should be a string, found <class 'list'>",
    ):
        sr.str.filter_tokens(3, replacement=["a", "b"])

    with pytest.raises(
        TypeError,
        match="Type of delimiter should be a string, found <class 'list'>",
    ):
        sr.str.filter_tokens(3, delimiter=["a", "b"])


def test_edit_distance():
    sr = cudf.Series(["kitten", "saturday", "address", "book"])
    tg = cudf.Series(["sitting", "sunday", "addressee", "back"])

    expected = cudf.Series([3, 3, 2, 2], dtype=np.int32)
    actual = sr.str.edit_distance(tg)
    assert_eq(expected, actual)

    expected = cudf.Series([0, 7, 6, 6], dtype=np.int32)
    actual = sr.str.edit_distance("kitten")
    assert_eq(expected, actual)


def test_edit_distance_matrix():
    # normal
    sr = cudf.Series(["rounded", "bounded", "bounce", "trounce", "ounce"])

    expected = cudf.Series(
        [
            [0, 1, 3, 3, 3],
            [1, 0, 2, 4, 3],
            [3, 2, 0, 2, 1],
            [3, 4, 2, 0, 2],
            [3, 3, 1, 2, 0],
        ]
    )
    got = sr.str.edit_distance_matrix()

    assert_eq(expected, got, check_dtype=False)

    # 1-row series
    sr2 = cudf.Series(["x"])
    with pytest.raises(ValueError, match="Require size >= 2"):
        sr2.str.edit_distance_matrix()

    # null rows
    sr3 = cudf.Series(["rounded", None, "bounce", "trounce", "ounce"])
    with pytest.raises(ValueError, match="Cannot compute"):
        sr3.str.edit_distance_matrix()


def test_porter_stemmer_measure():
    strings = cudf.Series(
        [
            "tr",
            "ee",
            "tree",
            "y",
            "by",
            "trouble",
            "oats",
            "trees",
            "ivy",
            "troubles",
            "private",
            "oaten",
            "orrery",
            None,
            "",
        ]
    )
    expected = cudf.Series(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, None, 0], dtype=np.int32
    )

    actual = strings.str.porter_stemmer_measure()

    assert type(expected) == type(actual)
    assert_eq(expected, actual)


def test_is_vowel_consonant():
    strings = cudf.Series(
        ["tr", "ee", "tree", "y", "by", "oats", "ivy", "orrery", None, ""]
    )
    expected = cudf.Series(
        [False, False, True, False, False, False, True, False, None, False]
    )
    actual = strings.str.is_vowel(2)
    assert type(expected) == type(actual)
    assert_eq(expected, actual)

    expected = cudf.Series(
        [True, False, True, False, False, False, True, True, None, False]
    )
    actual = strings.str.is_consonant(1)
    assert type(expected) == type(actual)
    assert_eq(expected, actual)

    indices = cudf.Series([2, 1, 0, 0, 1, 2, 0, 3, 0, 0])
    expected = cudf.Series(
        [False, True, False, False, True, False, True, True, None, False]
    )
    actual = strings.str.is_vowel(indices)
    assert type(expected) == type(actual)
    assert_eq(expected, actual)

    expected = cudf.Series(
        [False, False, True, True, False, True, False, False, None, False]
    )
    actual = strings.str.is_consonant(indices)
    assert type(expected) == type(actual)
    assert_eq(expected, actual)


def test_minhash():
    strings = cudf.Series(["this is my", "favorite book", None, ""])
    expected = cudf.Series([21141582, 962346254, None, 0], dtype=np.uint32)
    actual = strings.str.minhash()
    assert_eq(expected, actual)
    seeds = cudf.Series([0, 1, 2], dtype=np.uint32)
    expected = cudf.Series(
        [
            cudf.Series([1305480167, 668155704, 34311509], dtype=np.uint32),
            cudf.Series([32665384, 3470118, 363147162], dtype=np.uint32),
            None,
            cudf.Series([0, 0, 0], dtype=np.uint32),
        ]
    )
    actual = strings.str.minhash(seeds=seeds, n=5)
    assert_eq(expected, actual)

    with pytest.raises(ValueError):
        strings.str.minhash(seeds=7)
    with pytest.raises(ValueError):
        strings.str.minhash(seeds=seeds, method="md5")
    with pytest.raises(ValueError):
        seeds = cudf.Series([0, 1, 2], dtype=np.int32)
        strings.str.minhash(seeds=seeds)
