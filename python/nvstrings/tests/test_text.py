# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np
import nvstrings
import nvtext

from librmm_cffi import librmm as rmm


def test_tokenize():
    # default space delimiter
    strs = nvstrings.to_device(
        ["the quick fox jumped over the lazy dog",
         "the siamésé cat jumped under the sofa",
         None,
         ""]
    )
    outcome = nvtext.tokenize(strs)
    expected = ["the", "quick", "fox", "jumped", "over", "the", "lazy", "dog",
                "the", "siamésé", "cat", "jumped", "under", "the", "sofa"]
    assert outcome.to_host() == expected


def test_token_count():
    # default space delimiter
    strs = nvstrings.to_device(
        ["the quick brown fox jumped over the lazy brown dog",
         "the sable siamésé cat jumped under the brown sofa",
         None,
         ""]
    )
    outcome = nvtext.token_count(strs)
    expected = [10, 9, 0, 0]
    assert outcome == expected

    # custom delimiter
    outcome = nvtext.token_count(strs, delimiter='o')
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
        ["this is my favorite book",
         "Your Favorite book is different",
         None,
         ""]
    )
    unique_tokens_outcome = nvtext.unique_tokens(strs)
    expected = set(['Favorite', 'Your', 'book', 'different',
                    'favorite', 'is', 'my', 'this'])
    assert set(unique_tokens_outcome.to_host()) == expected

    # custom delimiter
    unique_tokens_outcome = nvtext.unique_tokens(strs, delimiter='my')
    expected = set([' favorite book', 'Your Favorite book is different',
                    'this is '])
    assert set(unique_tokens_outcome.to_host()) == expected


def test_contains_strings():
    strs = nvstrings.to_device(
        ["apples are green",
         "apples are a fruit",
         None,
         ""]
    )

    query_strings = nvstrings.to_device(['apple', 'fruit'])

    # host results
    contains_outcome = nvtext.contains_strings(strs, query_strings)
    expected = [
        [True, False],
        [True, True],
        [False, False],
        [False, False]
    ]
    assert contains_outcome == expected

    # device results
    outcome_darray = rmm.device_array((strs.size(), query_strings.size()),
                                      dtype=np.bool)
    nvtext.contains_strings(strs, query_strings,
                            devptr=outcome_darray.device_ctypes_pointer.value)
    assert np.array_equal(outcome_darray.copy_to_host(), expected)


def test_strings_counts():
    strs = nvstrings.to_device(
        ["apples are green",
         "apples are a fruit",
         None,
         ""]
    )

    query_strings = nvstrings.to_device(['pl', 're'])

    # host results
    contains_outcome = nvtext.strings_counts(strs, query_strings)
    expected = [
        [1, 2],
        [1, 1],
        [0, 0],
        [0, 0]
    ]
    assert contains_outcome == expected

    # device results
    outcome_darray = rmm.device_array((strs.size(), query_strings.size()),
                                      dtype=np.int32)
    nvtext.strings_counts(strs, query_strings,
                          devptr=outcome_darray.device_ctypes_pointer.value)
    assert np.array_equal(outcome_darray.copy_to_host(), expected)


def test_tokens_counts():
    strs = nvstrings.to_device(
        ["apples are green",
         "apples are a fruit",
         None,
         ""]
    )

    query_strings = nvtext.unique_tokens(strs)

    # host results
    contains_outcome = nvtext.tokens_counts(strs, query_strings)
    expected = [
        [0, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    assert contains_outcome == expected

    # device results
    outcome_darray = rmm.device_array((strs.size(), query_strings.size()),
                                      dtype=np.int32)
    nvtext.tokens_counts(strs, query_strings,
                         devptr=outcome_darray.device_ctypes_pointer.value)
    assert np.array_equal(outcome_darray.copy_to_host(), expected)


def test_edit_distance():
    # singe comparator
    strs = nvstrings.to_device(["my favorite sentence", "kittin", "nvidia"])
    distance_outcomes = nvtext.edit_distance(strs, 'kitten', algo=0)
    expected = [15, 1, 6]
    assert distance_outcomes == expected

    # multiple comparators
    comparators = nvstrings.to_device(
        ['my least favorite sentence', 'fish', 'software'])
    distance_outcomes = nvtext.edit_distance(strs, comparators, algo=0)
    expected = [6, 5, 7]
    assert distance_outcomes == expected


def test_ngrams():
    # bigrams
    strings = ['this is my favorite', 'book on my bookshelf']
    dstrings = nvstrings.to_device(strings)
    expected = [
        'this_is',
        'is_my',
        'my_favorite',
        'favorite_book',
        'book_on',
        'on_my',
        'my_bookshelf'
    ]
    tokens = nvtext.tokenize(dstrings)
    outcome = nvtext.ngrams(tokens, N=2, sep='_')
    assert outcome.to_host() == expected

    # trigrams
    strings = ['this is my favorite', 'book on my bookshelf']
    dstrings = nvstrings.to_device(strings)
    expected = [
        'this-is-my',
        'is-my-favorite',
        'my-favorite-book',
        'favorite-book-on',
        'book-on-my',
        'on-my-bookshelf'
    ]
    tokens = nvtext.tokenize(dstrings)
    outcome = nvtext.ngrams(tokens, N=3, sep='-')
    assert outcome.to_host() == expected
