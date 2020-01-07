# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np

import rmm

import nvstrings
import nvtext


def test_tokenize():
    # default space delimiter
    strs = nvstrings.to_device(
        [
            "the quick fox jumped over the lazy dog",
            "the siamésé cat jumped under the sofa",
            None,
            "",
        ]
    )
    outcome = nvtext.tokenize(strs)
    expected = [
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
    assert outcome.to_host() == expected


def test_token_count():
    # default space delimiter
    strs = nvstrings.to_device(
        [
            "the quick brown fox jumped over the lazy brown dog",
            "the sable siamésé cat jumped under the brown sofa",
            None,
            "",
        ]
    )
    outcome = nvtext.token_count(strs)
    expected = [10, 9, 0, 0]
    assert outcome == expected

    # custom delimiter
    outcome = nvtext.token_count(strs, delimiter="o")
    expected = [6, 3, 0, 0]
    assert outcome == expected

    # test device pointer
    outcome_darray = rmm.device_array(strs.size(), dtype=np.int32)
    nvtext.token_count(strs, devptr=outcome_darray.device_ctypes_pointer.value)
    expected = [10, 9, 0, 0]
    assert np.array_equal(outcome_darray.copy_to_host(), expected)


def test_unique_tokens():
    # default space delimiter
    strs = nvstrings.to_device(
        [
            "this is my favorite book",
            "Your Favorite book is different",
            None,
            "",
        ]
    )
    unique_tokens_outcome = nvtext.unique_tokens(strs)
    expected = set(
        [
            "Favorite",
            "Your",
            "book",
            "different",
            "favorite",
            "is",
            "my",
            "this",
        ]
    )
    assert set(unique_tokens_outcome.to_host()) == expected

    # custom delimiter
    unique_tokens_outcome = nvtext.unique_tokens(strs, delimiter="my")
    expected = set(
        [" favorite book", "Your Favorite book is different", "this is "]
    )
    assert set(unique_tokens_outcome.to_host()) == expected


def test_character_tokenize():
    strs = nvstrings.to_device(
        [
            "the quick fox jumped over the lazy dog",
            "the siamésé cat jumped under the sofa",
            None,
            "",
        ]
    )
    outcome = nvtext.character_tokenize(strs)
    expected = [
        "t",
        "h",
        "e",
        " ",
        "q",
        "u",
        "i",
        "c",
        "k",
        " ",
        "f",
        "o",
        "x",
        " ",
        "j",
        "u",
        "m",
        "p",
        "e",
        "d",
        " ",
        "o",
        "v",
        "e",
        "r",
        " ",
        "t",
        "h",
        "e",
        " ",
        "l",
        "a",
        "z",
        "y",
        " ",
        "d",
        "o",
        "g",
        "t",
        "h",
        "e",
        " ",
        "s",
        "i",
        "a",
        "m",
        "é",
        "s",
        "é",
        " ",
        "c",
        "a",
        "t",
        " ",
        "j",
        "u",
        "m",
        "p",
        "e",
        "d",
        " ",
        "u",
        "n",
        "d",
        "e",
        "r",
        " ",
        "t",
        "h",
        "e",
        " ",
        "s",
        "o",
        "f",
        "a",
    ]

    assert outcome.to_host() == expected


def test_contains_strings():
    strs = nvstrings.to_device(
        ["apples are green", "apples are a fruit", None, ""]
    )

    query_strings = nvstrings.to_device(["apple", "fruit"])

    # host results
    contains_outcome = nvtext.contains_strings(strs, query_strings)
    expected = [[True, False], [True, True], [False, False], [False, False]]
    assert contains_outcome == expected

    # device results
    outcome_darray = rmm.device_array(
        (strs.size(), query_strings.size()), dtype=np.bool
    )
    nvtext.contains_strings(
        strs, query_strings, devptr=outcome_darray.device_ctypes_pointer.value
    )
    assert np.array_equal(outcome_darray.copy_to_host(), expected)


def test_strings_counts():
    strs = nvstrings.to_device(
        ["apples are green", "apples are a fruit", None, ""]
    )

    query_strings = nvstrings.to_device(["pl", "re"])

    # host results
    contains_outcome = nvtext.strings_counts(strs, query_strings)
    expected = [[1, 2], [1, 1], [0, 0], [0, 0]]
    assert contains_outcome == expected

    # device results
    outcome_darray = rmm.device_array(
        (strs.size(), query_strings.size()), dtype=np.int32
    )
    nvtext.strings_counts(
        strs, query_strings, devptr=outcome_darray.device_ctypes_pointer.value
    )
    assert np.array_equal(outcome_darray.copy_to_host(), expected)


def test_tokens_counts():
    strs = nvstrings.to_device(
        ["apples are green", "apples are a fruit", None, ""]
    )

    query_strings = nvtext.unique_tokens(strs)

    # host results
    contains_outcome = nvtext.tokens_counts(strs, query_strings)
    expected = [
        [0, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert contains_outcome == expected

    # device results
    outcome_darray = rmm.device_array(
        (strs.size(), query_strings.size()), dtype=np.int32
    )
    nvtext.tokens_counts(
        strs, query_strings, devptr=outcome_darray.device_ctypes_pointer.value
    )
    assert np.array_equal(outcome_darray.copy_to_host(), expected)


def test_replace_tokens():
    strs = nvstrings.to_device(
        [
            "the quick fox jumped over the lazy dog",
            "the siamésé cat jumped under the sofa",
            None,
            "",
        ]
    )
    tokens = nvstrings.to_device(["the", "over", "under"])
    repls = nvstrings.to_device(["1", "2", "3"])
    outcome = nvtext.replace_tokens(strs, tokens, repls)
    expected = [
        "1 quick fox jumped 2 1 lazy dog",
        "1 siamésé cat jumped 3 1 sofa",
        None,
        "",
    ]
    assert outcome.to_host() == expected


def test_normalize_spaces():
    strs = nvstrings.to_device(
        [
            " the\t quick fox  jumped over the lazy dog",
            "the siamésé cat\f jumped\t\tunder the sofa  ",
            None,
            "",
        ]
    )
    outcome = nvtext.normalize_spaces(strs)
    expected = [
        "the quick fox jumped over the lazy dog",
        "the siamésé cat jumped under the sofa",
        None,
        "",
    ]
    assert outcome.to_host() == expected


def test_edit_distance():
    # singe comparator
    strs = nvstrings.to_device(["my favorite sentence", "kittin", "nvidia"])
    distance_outcomes = nvtext.edit_distance(strs, "kitten", algo=0)
    expected = [15, 1, 6]
    assert distance_outcomes == expected

    # multiple comparators
    comparators = nvstrings.to_device(
        ["my least favorite sentence", "fish", "software"]
    )
    distance_outcomes = nvtext.edit_distance(strs, comparators, algo=0)
    expected = [6, 5, 7]
    assert distance_outcomes == expected


def test_edit_distance_matrix():

    strs = nvstrings.to_device(
        ["my least favorite sentence", "fish", "software"]
    )
    distance_outcomes = nvtext.edit_distance_matrix(strs, algo=0)
    expected = [[0, 23, 22], [23, 0, 7], [22, 7, 0]]
    assert distance_outcomes == expected


def test_ngrams():
    # bigrams
    strings = ["this is my favorite", "book on my bookshelf"]
    dstrings = nvstrings.to_device(strings)
    expected = [
        "this_is",
        "is_my",
        "my_favorite",
        "favorite_book",
        "book_on",
        "on_my",
        "my_bookshelf",
    ]
    tokens = nvtext.tokenize(dstrings)
    outcome = nvtext.ngrams(tokens, N=2, sep="_")
    assert outcome.to_host() == expected

    # trigrams
    strings = ["this is my favorite", "book on my bookshelf"]
    dstrings = nvstrings.to_device(strings)
    expected = [
        "this-is-my",
        "is-my-favorite",
        "my-favorite-book",
        "favorite-book-on",
        "book-on-my",
        "on-my-bookshelf",
    ]
    tokens = nvtext.tokenize(dstrings)
    outcome = nvtext.ngrams(tokens, N=3, sep="-")
    assert outcome.to_host() == expected


def test_scatter_count():
    # regular
    strings = ["Dickens", "Einstein", "Christie"]
    dstrings = nvstrings.to_device(strings)
    expected = [
        "Dickens",
        "Einstein",
        "Einstein",
        "Christie",
        "Christie",
        "Christie",
    ]
    outcome = nvtext.scatter_count(dstrings, [1, 2, 3])
    assert outcome.to_host() == expected

    # with input as GPU mem pointer
    arr = np.array([1, 2, 3], dtype="int32")
    dev_arr = rmm.to_device(arr)
    got = nvtext.scatter_count(dstrings, dev_arr.device_ctypes_pointer.value)
    assert got.to_host() == expected

    # with nulls
    expected = ["Dickens", "Dickens"]
    outcome = nvtext.scatter_count(dstrings, [2, 0, None])
    assert outcome.to_host() == expected
