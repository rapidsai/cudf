# Copyright (c) 2019, NVIDIA CORPORATION.

import nvstrings
import nvtext

from utils import assert_eq

hstrs = ["the quick brown fox jumps over the lazy dog",
         "the fat cat lays next to the other accénted cat",
         "a slow moving turtlé cannot catch the bird",
         "", None]

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves",
              "you", "your", "yours", "yourself", "yourselves", "he", "him",
              "his", "himself", "she", "her", "hers", "herself", "it", "its",
              "itself", "they", "them", "their", "theirs", "themselves",
              "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been",
              "being", "have", "has", "had", "having", "do", "does", "did",
              "doing", "a", "an", "the", "and", "but", "if", "or", "because",
              "as", "until", "while", "of", "at", "by", "for", "with",
              "about", "against", "between", "into", "through", "during",
              "before", "after", "above", "below", "to", "from", "up",
              "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where",
              "why", "how", "all", "any", "both", "each", "few", "more",
              "most", "other", "some", "such", "no", "nor", "not", "only",
              "own", "same", "so", "than", "too", "very", "s", "t", "can",
              "will", "just", "don", "should", "now", "uses", "use", "using",
              "used", "one", "also"]


def test_replace():
    nvstrs = nvstrings.to_device(hstrs)
    nvtgts = nvstrings.to_device(['the ', 'a ', 'in '])
    got = nvstrs.replace_multi(nvtgts, ' ', regex=False)
    expected = [' quick brown fox jumps over  lazy dog',
                ' fat cat lays next to  other accénted cat',
                ' slow moving turtlé cannot catch  bird',
                '', None]
    assert_eq(got, expected)

    nvtgts = nvstrings.to_device([' dog', ' cat', ' bird'])
    nvrpls = nvstrings.to_device([' DOG', ' CAT', ' BIRD'])
    got = nvstrs.replace_multi(nvtgts, nvrpls, regex=False)
    expected = ['the quick brown fox jumps over the lazy DOG',
                'the fat CAT lays next to the other accénted CAT',
                'a slow moving turtlé cannot CATch the BIRD',
                '', None]
    assert_eq(got, expected)


def test_replace_re():
    nvstrs = nvstrings.to_device(hstrs)
    stop_words_re = []
    for w in stop_words:
        stop_words_re.append('\\b' + w + '\\b')
    got = nvstrs.replace_multi(stop_words_re, [''])
    expected = [' quick brown fox jumps   lazy dog',
                ' fat cat lays next    accénted cat',
                ' slow moving turtlé cannot catch  bird',
                '', None]
    assert_eq(got, expected)


def test_replace_tokens():
    nvstrs = nvstrings.to_device(hstrs)
    tokens = nvstrings.to_device(stop_words)
    got = nvtext.replace_tokens(nvstrs, tokens, '')
    expected = [' quick brown fox jumps   lazy dog',
                ' fat cat lays next    accénted cat',
                ' slow moving turtlé cannot catch  bird',
                '', None]
    assert_eq(got, expected)
