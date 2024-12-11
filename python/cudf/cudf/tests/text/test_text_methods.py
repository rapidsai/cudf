# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import random
import string

import numpy as np
import pytest

import cudf
from cudf.core.byte_pair_encoding import BytePairEncoder
from cudf.core.tokenize_vocabulary import TokenizeVocabulary
from cudf.testing import assert_eq


def test_tokenize():
    strings = cudf.Series(
        [
            "the quick fox jumped over the lazy dog",
            "the siamésé cat jumped under the sofa",
            None,
            "",
        ]
    )

    expected_values = [
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

    expected_values = [
        "the quick f",
        "x jumped ",
        "ver the lazy d",
        "g",
        "the siamésé cat jumped under the s",
        "fa",
    ]
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


@pytest.mark.parametrize(
    "delimiter, input, default_id, results",
    [
        (
            "",
            "the quick brown fox jumps over the lazy brown dog",
            99,
            [0, 1, 2, 3, 4, 5, 0, 99, 2, 6],
        ),
        (
            " ",
            " the sable siamésé cat jumps under the brown sofa ",
            -1,
            [0, 7, 8, 9, 4, 10, 0, 2, 11],
        ),
        (
            "_",
            "the_quick_brown_fox_jumped__over_the_lazy_brown_dog",
            -99,
            [0, 1, 2, 3, -99, 5, 0, -99, 2, 6],
        ),
    ],
)
def test_tokenize_with_vocabulary(delimiter, input, default_id, results):
    vocabulary = cudf.Series(
        [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "dog",
            "sable",
            "siamésé",
            "cat",
            "under",
            "sofa",
        ]
    )
    tokenizer = TokenizeVocabulary(vocabulary)

    strings = cudf.Series([input, None, "", input])

    expected = cudf.Series(
        [
            cudf.Series(results, dtype=np.int32),
            None,
            cudf.Series([], dtype=np.int32),
            cudf.Series(results, dtype=np.int32),
        ]
    )

    actual = tokenizer.tokenize(strings, delimiter, default_id)
    assert type(expected) == type(actual)
    assert_eq(expected, actual)


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
            ],
            [1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5],
            False,
        ),
        (
            3,
            [
                "thi",
                "his",
                "boo",
                "ook",
                "her",
                "ere",
            ],
            [1, 1, 4, 4, 5, 5],
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


def test_hash_character_ngrams():
    strings = cudf.Series(["abcdefg", "stuvwxyz"])
    expected = cudf.Series(
        [
            cudf.Series([3902511862, 570445242, 4202475763], dtype=np.uint32),
            cudf.Series(
                [556054766, 3166857694, 3760633458, 192452857], dtype=np.uint32
            ),
        ]
    )
    actual = strings.str.hash_character_ngrams(5, True)
    assert type(expected) == type(actual)
    assert_eq(expected, actual)

    actual = strings.str.hash_character_ngrams(5)
    expected = expected.explode()
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
            (
                "goodbye, one-two:three~four+five_six@sev"
                "en#eight^nine heŒŽ‘•™œ$µ¾ŤƠé Ǆ"
            ),
        ]
    )
    expected_values = [
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
    sr = cudf.Index(
        [
            "hello world",
            "sdf",
            (
                "goodbye, one-two:three~four+five_six@sev"
                "en#eight^nine heŒŽ‘•™œ$µ¾ŤƠé Ǆ"
            ),
        ]
    )
    expected = cudf.Index(
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

    sr = cudf.Index(["a"])
    expected = cudf.Index(["a"])

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


def test_minhash_permuted():
    strings = cudf.Series(["this is my", "favorite book", None, ""])

    params = cudf.Series([1, 2, 3], dtype=np.uint32)
    expected = cudf.Series(
        [
            cudf.Series([1305480168, 462824406, 74608229], dtype=np.uint32),
            cudf.Series([32665385, 65330770, 97996155], dtype=np.uint32),
            None,
            cudf.Series([0, 0, 0], dtype=np.uint32),
        ]
    )
    actual = strings.str.minhash_permuted(0, a=params, b=params, width=5)
    assert_eq(expected, actual)

    params = cudf.Series([1, 2, 3], dtype=np.uint64)
    expected = cudf.Series(
        [
            cudf.Series(
                [105531920695060180, 172452388517576009, 316595762085180524],
                dtype=np.uint64,
            ),
            cudf.Series(
                [35713768479063122, 71427536958126236, 58787297728258212],
                dtype=np.uint64,
            ),
            None,
            cudf.Series([0, 0, 0], dtype=np.uint64),
        ]
    )
    actual = strings.str.minhash64_permuted(0, a=params, b=params, width=5)
    assert_eq(expected, actual)

    # test wrong seed types
    with pytest.raises(ValueError):
        strings.str.minhash_permuted(1, a="a", b="b", width=7)
    with pytest.raises(ValueError):
        params = cudf.Series([0, 1, 2], dtype=np.int32)
        strings.str.minhash_permuted(1, a=params, b=params, width=6)
    with pytest.raises(ValueError):
        params = cudf.Series([0, 1, 2], dtype=np.uint32)
        strings.str.minhash64_permuted(1, a=params, b=params, width=8)


def test_word_minhash():
    ls = cudf.Series([["this", "is", "my"], ["favorite", "book"]])

    expected = cudf.Series(
        [
            cudf.Series([21141582], dtype=np.uint32),
            cudf.Series([962346254], dtype=np.uint32),
        ]
    )
    actual = ls.str.word_minhash()
    assert_eq(expected, actual)
    seeds = cudf.Series([0, 1, 2], dtype=np.uint32)
    expected = cudf.Series(
        [
            cudf.Series([21141582, 1232889953, 1268336794], dtype=np.uint32),
            cudf.Series([962346254, 2321233602, 1354839212], dtype=np.uint32),
        ]
    )
    actual = ls.str.word_minhash(seeds=seeds)
    assert_eq(expected, actual)

    expected = cudf.Series(
        [
            cudf.Series([2603139454418834912], dtype=np.uint64),
            cudf.Series([5240044617220523711], dtype=np.uint64),
        ]
    )
    actual = ls.str.word_minhash64()
    assert_eq(expected, actual)
    seeds = cudf.Series([0, 1, 2], dtype=np.uint64)
    expected = cudf.Series(
        [
            cudf.Series(
                [
                    2603139454418834912,
                    8644371945174847701,
                    5541030711534384340,
                ],
                dtype=np.uint64,
            ),
            cudf.Series(
                [5240044617220523711, 5847101123925041457, 153762819128779913],
                dtype=np.uint64,
            ),
        ]
    )
    actual = ls.str.word_minhash64(seeds=seeds)
    assert_eq(expected, actual)

    # test wrong seed types
    with pytest.raises(ValueError):
        ls.str.word_minhash(seeds="a")
    with pytest.raises(ValueError):
        seeds = cudf.Series([0, 1, 2], dtype=np.int32)
        ls.str.word_minhash(seeds=seeds)
    with pytest.raises(ValueError):
        seeds = cudf.Series([0, 1, 2], dtype=np.uint32)
        ls.str.word_minhash64(seeds=seeds)


def test_jaccard_index():
    str1 = cudf.Series(["the brown dog", "jumped about"])
    str2 = cudf.Series(["the black cat", "jumped around"])

    expected = cudf.Series([0.058824, 0.307692], dtype=np.float32)
    actual = str1.str.jaccard_index(str2, 5)
    assert_eq(expected, actual)

    actual = str2.str.jaccard_index(str1, 5)
    assert_eq(expected, actual)

    with pytest.raises(ValueError):
        str1.str.jaccard_index(str2, 1)
    with pytest.raises(ValueError):
        str3 = cudf.Series(["not enough rows"])
        str1.str.jaccard_index(str3, 5)


def _make_list_of_strings_of_random_length(
    num_strings, min_length, max_length
):
    return [
        "".join(
            random.choice(string.ascii_lowercase)
            for _ in range(random.randint(min_length, max_length))
        )
        for _ in range(num_strings)
    ]


def test_jaccard_index_random_strings():
    # Seed the rng before random string generation.
    random.seed(42)
    num_strings = 100
    jaccard_width = 5
    common_strings = _make_list_of_strings_of_random_length(
        num_strings, jaccard_width, 50
    )
    uncommon_strings1 = _make_list_of_strings_of_random_length(
        num_strings, jaccard_width, 10
    )
    uncommon_strings2 = _make_list_of_strings_of_random_length(
        num_strings, jaccard_width, 20
    )
    str1 = cudf.Series(uncommon_strings1).str.cat(cudf.Series(common_strings))
    str2 = cudf.Series(uncommon_strings2).str.cat(cudf.Series(common_strings))

    # adopted from https://github.com/rapidsai/rapids-deduplication/issues/36
    da = str1.str.character_ngrams(jaccard_width, True)
    db = str2.str.character_ngrams(jaccard_width, True)
    da = da.list.unique()
    db = db.list.unique()
    da = da.explode()
    db = db.explode()
    da = da.to_frame()
    db = db.to_frame()
    da = da.reset_index()
    db = db.reset_index()
    da = da.rename(columns={0: "token"})
    db = db.rename(columns={0: "token"})
    db["match"] = 1
    inter = da.merge(db, on=["index", "token"], how="left")
    inter = inter.groupby("index")["match"].sum()
    union = da.merge(db, on=["index", "token"], how="outer")
    union = union.groupby("index").size()
    res = inter / union
    res.fillna(0, inplace=True)
    res = res.sort_index()
    res = res.values.astype("float32")
    expected = cudf.Series(res)

    actual = str1.str.jaccard_index(str2, jaccard_width)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "separator, input, results",
    [
        (" ", "thetestsentence", "the test sent ence"),
        ("_", "sentenceistest", "sent_ence_is_test"),
        ("$", "istestsentencehere", "is$test$sent$ence$he$r$e"),
    ],
)
def test_byte_pair_encoding(separator, input, results):
    pairs_table = cudf.Series(
        [
            "t he",
            "h e",
            "e n",
            "i t",
            "i s",
            "e s",
            "en t",
            "c e",
            "es t",
            "en ce",
            "t h",
            "h i",
            "th is",
            "t est",
            "s i",
            "s ent",
        ]
    )
    encoder = BytePairEncoder(pairs_table)

    strings = cudf.Series([input, None, "", input])

    expected = cudf.Series([results, None, "", results])

    actual = encoder(strings, separator)
    assert type(expected) == type(actual)
    assert_eq(expected, actual)
