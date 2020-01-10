# Copyright (c) 2019, NVIDIA CORPORATION.

import nvstrings as nvs
import pyniNVText


def tokenize(strs, delimiter=None):
    """
    Each string is split into tokens using the provided delimiter(s).
    The nvstrings instance returned contains the tokens in the order
    they were found.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation
    delimiter : str or nvstrings or list of strs
        The string used to locate the split points of each string.
        Default is whitespace.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello world",
    ...                          "goodbye world",
    ...                          "hello goodbye"])
    >>> t = nvtext.tokenize(s)
    >>> print(t)
    ["hello","world","goodbye","world","hello","goodbye"]

    """
    rtn = None
    if delimiter is None:
        rtn = pyniNVText.n_tokenize(strs, delimiter)
    if isinstance(delimiter, str):
        rtn = pyniNVText.n_tokenize(strs, delimiter)
    if isinstance(delimiter, list):
        delimiter = nvs.to_device(delimiter)
    if isinstance(delimiter, nvs.nvstrings):
        rtn = pyniNVText.n_tokenize_multi(strs, delimiter)
    if rtn is not None:
        rtn = nvs.nvstrings(rtn)
    return rtn


def unique_tokens(strs, delimiter=None):
    """
    Each string is split into tokens using the provided delimiter.
    The nvstrings instance returned contains unique list of tokens.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation
    delimiter : str
        The character used to locate the split points of each string.
        Default is whitespace.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello world",
    ...                          "goodbye world",
    ...                          "hello goodbye"])
    >>> ut = nvtext.unique_tokens(s)
    >>> print(ut)
    ["goodbye","hello","world"]

    """
    rtn = pyniNVText.n_unique_tokens(strs, delimiter)
    if rtn is not None:
        rtn = nvs.nvstrings(rtn)
    return rtn


def token_count(strs, delimiter=None, devptr=0):
    """
    Each string is split into tokens using the provided delimiter.
    The returned integer array is the number of tokens in each string.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation
    delimiter : str
        The character used to locate the split points of each string.
        Default is whitespace.
    devptr : GPU memory pointer
        Must be able to hold at least strs.size() of int32 values.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello world","goodbye",""])
    >>> n = nvtext.token_count(s)
    >>> print(n)
    [2,1,0]

    """
    rtn = pyniNVText.n_token_count(strs, delimiter, devptr)
    return rtn


def character_tokenize(strs):
    """
    Each string is split into individual characters.
    The nvstrings instance returned contains each character as an
    individual string.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello world",
    ...                          "goodbye"])
    >>> t = nvtext.character_tokenize(s)
    >>> print(t)
    ["h","e","l","l","o"," ","w","o","r","l","d","g","o","o","d","b","y","e"]

    """
    rtn = pyniNVText.n_character_tokenize(strs)
    if rtn is not None:
        rtn = nvs.nvstrings(rtn)
    return rtn


def contains_strings(strs, tgts, devptr=0):
    """
    The tgts strings are searched for within each strs.
    The returned byte array is 1 for each tgts in strs and 0 otherwise.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation.
    tgts : nvstrings
        The strings to check for inside each strs.
    devptr : GPU memory pointer
        Must be able to hold at least strs.size()*tgts.size()
        of int8 values.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello","goodbye",""])
    >>> t = nvstrings.to_device(['o','y'])
    >>> n = nvtext.contains_strings(s,t)
    >>> print(n)
    [[True,False],[True,True],[False,False]]

    """
    rtn = pyniNVText.n_contains_strings(strs, tgts, devptr)
    return rtn


def strings_counts(strs, tgts, devptr=0):
    """
    The tgts strings are searched for within each strs.
    The returned int32 array is number of occurrences of each tgts in strs.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation.
    tgts : nvstrings
        The strings to count for inside each strs.
    devptr : GPU memory pointer
        Must be able to hold at least strs.size()*tgts.size()
        of int32 values.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello","goodbye",""])
    >>> t = nvstrings.to_device(['o','y'])
    >>> n = nvtext.strings_counts(s,t)
    >>> print(n)
    [[1,0],[2,1],[0,0]]

    """
    rtn = pyniNVText.n_strings_counts(strs, tgts, devptr)
    return rtn


def tokens_counts(strs, tgts, delimiter=None, devptr=0):
    """
    The tgts strings are searched for within each strs.
    The returned int32 array is number of occurrences of each tgts in strs.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation.
    tgts : nvstrings
        The strings to count for inside each strs.
    delimiter : str
        The character used to locate the split points of each string.
        Default is whitespace.
    devptr : GPU memory pointer
        Must be able to hold at least strs.size()*tgts.size()
        of int32 values.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["this is me","goodbye me",""])
    >>> t = nvstrings.to_device(['is','me'])
    >>> n = nvtext.tokens_counts(s,t)
    >>> print(n)
    [[1,1],[0,1],[0,0]]

    """
    rtn = pyniNVText.n_tokens_counts(strs, tgts, delimiter, devptr)
    return rtn


def replace_tokens(strs, tgts, repls, delimiter=None):
    """
    The tgts tokens are searched for within each strs and replaced
    with the corresponding repls if found. Tokens are identified by
    the delimiter character(s) provided.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation.
    tgts : nvstrings
        The tokens to search for inside each strs.
    repls : nvstrings or str
        The strings to replace for each found tgts token found.
        Alternately, this can be a single str instance
        and would be used as replacement for each string found.
    delimiter : str
        The characters used to locate the tokens of each string.
        Default is whitespace.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["this is me","theme music",""])
    >>> t = nvstrings.to_device(['is','me'])
    >>> r = nvtext.replace_tokens(s,t,'_')
    >>> print(r)
    ["this _ _", "theme music"]

    """
    if isinstance(repls, str):
        repls = nvs.to_device([repls])
    if isinstance(repls, list):
        repls = nvs.to_device(repls)
    if isinstance(tgts, list):
        tgts = nvs.to_device(tgts)
    rtn = pyniNVText.n_replace_tokens(strs, tgts, repls, delimiter)
    if rtn is not None:
        rtn = nvs.nvstrings(rtn)
    return rtn


def normalize_spaces(strs):
    """
    Remove extra whitespace between tokens and trim whitespace
    from the beginning and the end of each string.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["hello \t world"," test string  "])
    >>> n = nvtext.normalize_spaces(s)
    >>> print(n)
    ["hello world", "test string"]

    """
    rtn = pyniNVText.n_normalize_spaces(strs)
    if rtn is not None:
        rtn = nvs.nvstrings(rtn)
    return rtn


def edit_distance(strs, tgt, algo=0, devptr=0):
    """
    Compute the edit-distance between strs and tgt.
    Edit distance is how many character changes between strings.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation.
    tgt : str, nvstrings
        The string or strings to compute edit-distance with.
    algo: int
        0 = Levenshtein
    devptr : GPU memory pointer
        Must be able to hold at least strs.size() of int32 values.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["honda","hyundai"])
    >>> n = nvtext.edit_distance(s,"honda")
    >>> print(n)
    [0,3]

    """
    rtn = pyniNVText.n_edit_distance(strs, tgt, algo, devptr)
    return rtn


def edit_distance_matrix(strs, algo=0, devptr=0):
    """
    Compute the edit-distance between each pair of strings in strs .
    Edit distance is how many character changes between strings.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation.
    algo: int
        0 = Levenshtein
    devptr : GPU memory pointer
        Must be able to hold at least strs.size()*strs.size() of int32 values.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["honda","hyundai","suzuki"])
    >>> n = nvtext.edit_distance_matrix(s)
    >>> print(n)
    [[0, 3, 6],
     [3, 0, 5],
     [6, 5, 0]]

    """
    if strs.size() < 2:
        raise ValueError("minimum two strings are required!")
    if algo != 0:
        raise ValueError("Only `levenshtein` algorithm is supported yet.")
    rtn = pyniNVText.n_edit_distance_matrix(strs, algo, devptr)
    return rtn


def ngrams(tokens, N=2, sep="_"):
    """
    Generate the n-grams from a set of tokens.
    You can generate tokens from an nvstrings instance using
    the tokenize() function.

    Parameters
    ----------
    tokens : nvstrings
        The tokens for this operation.
    N : int
        The degree of the n-gram (number of consecutive tokens).
        Default of 2 for bigrams.
    sep : str
        The separator to use between within an n-gram.
        Default is '_'.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> dstrings = nvstrings.to_device(['this is my', 'favorite book'])
    >>> print(nvtext.ngrams(dstrings, N=2, sep='_'))
    ['this_is', 'is_my', 'my_favorite', 'favorite_book']
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    rtn = pyniNVText.n_create_ngrams(tokens, N, sep)
    if rtn is not None:
        rtn = nvs.nvstrings(rtn)
    return rtn


def ngrams_tokenize(strs, delimiter=" ", N=2, sep="_"):
    """
    Generate the n-grams using tokens from each string.
    This will tokenize each string and then generate ngrams for each string.

    Parameters
    ----------
    strs : nvstrings
        The tokens for this operation.
    delimiter : str
        The character used to locate the split points of each string.
        Default is space.
    N : int
        The degree of the n-gram (number of consecutive tokens).
        Default of 2 for bigrams.
    sep : str
        The separator to use between tokens within an n-gram.
        Default is '_'.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> dstrings = nvstrings.to_device(['this is the', 'best book'])
    >>> print(nvtext.ngrams_tokenize(dstrings, N=2, sep='_'))
    ['this_is', 'is_the', 'best_book']
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    rtn = pyniNVText.n_ngrams_tokenize(strs, delimiter, N, sep)
    if rtn is not None:
        rtn = nvs.nvstrings(rtn)
    return rtn


def scatter_count(strs, counts):
    """
    Create a new strings instance by duplicating each string by
    the count specified in counts.

    Parameters
    ----------
    strs : nvstrings
        The strings used for this operation.
    counts : list or GPU memory pointer
        Should be strs.size() of int32 values.
        Each value is the number of times the corresponding
        string is duplicated in the output instance.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> dstrings = nvstrings.to_device(['aaa', 'bbb', 'ccc'])
    >>> print(nvtext.scatter_count(dstrings, [1,2,3]))
    ['aaa', 'bbb', 'bbb', 'ccc', 'ccc', 'ccc']
    """
    if counts is None:
        raise ValueError("counts must not be None")
    rtn = pyniNVText.n_scatter_count(strs, counts)
    if rtn is not None:
        rtn = nvs.nvstrings(rtn)
    return rtn


def porter_stemmer_measure(strs, vowels="aeiou", y_char="y", devptr=0):
    """
    Calculates porter stemmer measure of each string

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation

    vowels: str
     chars to consider as vowels
     Default are 'aeiou'

    y_char: str
     characters to treat as 'y_char'
     Default is 'y'

    devptr : GPU memory pointer
        Must be able to hold at least strs.size() of int32 values.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["tr","trouble", "troubles"])
    >>> n = nvtext.porter_stemmer_measure(s)
    >>> print(n)
    >>> [0, 1, 2]
    """
    if not isinstance(strs, nvs.nvstrings):
        raise ValueError("strs must be nvstrings object")

    rtn = pyniNVText.n_porter_stemmer_measure(strs, vowels, y_char, devptr)
    return rtn


def is_consonant(
    strs, index, index_is_ptr=False, vowels="aeiou", y_char="y", devptr=0
):
    """
    Returns a bool array with True if the character at the specified
    (0-based) index position is a consonant.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation

    index : int
        The position of character to check in each string.
        This may also be a device pointer to memory containing
        index values of type int32 for each of the strings.
        The number of values must be at least strs.size().

    index_is_ptr : boolean
        Set to True if index parameter is device memory pointer
        instead of an character position.

    vowels: str
        Characters to consider as vowels
        Default is 'aeiou'

    y_char: str
        Characters to treat as 'y' for vowel-specific algorithm.
        Default is 'y'

    devptr : GPU memory pointer
        Must be able to hold at least strs.size() of int8 values.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["one", "berry", "blast"])
    >>> n = nvtext.is_consonant(s,4)
    >>> print(n)
    >>> [false, false, true]
    """
    if not isinstance(strs, nvs.nvstrings):
        raise ValueError("strs must be nvstrings object")

    rtn = pyniNVText.n_is_letter(
        strs, False, index, index_is_ptr, vowels, y_char, devptr
    )
    return rtn


def is_vowel(
    strs, index, index_is_ptr=False, vowels="aeiou", y_char="y", devptr=0
):
    """
    Returns a bool array with True if the character at the specified
    (0-based) index position is a vowel.

    Parameters
    ----------
    strs : nvstrings
        The strings for this operation.

    index : int
        The position of character to check in each string.
        This may also be a device pointer to memory containing
        index values of type int32 for each of the strings.
        The number of values must be at least strs.size().

    index_is_ptr : boolean
        Set to True if index parameter is device memory pointer
        instead of an character position.

    vowels: str
        Characters to consider as vowels
        Default is 'aeiou'

    y_char: str
        Characters to treat as 'y' for vowel-specific algorithm.
        Default is 'y'

    devptr : GPU memory pointer
        Must be able to hold at least strs.size() of int8 values.

    Examples
    --------
    >>> import nvstrings, nvtext
    >>> s = nvstrings.to_device(["twelve", "berry", "blazing", "two"])
    >>> n = nvtext.is_consonant(s,4)
    >>> print(n)
    >>> [false, true, true, false]
    """
    if not isinstance(strs, nvs.nvstrings):
        raise ValueError("strs must be nvstrings object")

    rtn = pyniNVText.n_is_letter(
        strs, True, index, index_is_ptr, vowels, y_char, devptr
    )
    return rtn
