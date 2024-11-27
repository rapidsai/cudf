# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from cudf._lib.nvtext.edit_distance import edit_distance, edit_distance_matrix
from cudf._lib.nvtext.generate_ngrams import (
    generate_character_ngrams,
    generate_ngrams,
    hash_character_ngrams,
)
from cudf._lib.nvtext.jaccard import jaccard_index
from cudf._lib.nvtext.minhash import (
    minhash,
    minhash64,
    minhash64_permuted,
    minhash_permuted,
    word_minhash,
    word_minhash64,
)
from cudf._lib.nvtext.ngrams_tokenize import ngrams_tokenize
from cudf._lib.nvtext.normalize import normalize_characters, normalize_spaces
from cudf._lib.nvtext.replace import filter_tokens, replace_tokens
from cudf._lib.nvtext.stemmer import (
    LetterType,
    is_letter,
    is_letter_multi,
    porter_stemmer_measure,
)
from cudf._lib.nvtext.tokenize import (
    _count_tokens_column,
    _count_tokens_scalar,
    _tokenize_column,
    _tokenize_scalar,
    character_tokenize,
    detokenize,
    tokenize_with_vocabulary,
)
from cudf._lib.strings.convert.convert_fixed_point import to_decimal
from cudf._lib.strings.convert.convert_floats import is_float
from cudf._lib.strings.convert.convert_integers import is_integer
from cudf._lib.strings.convert.convert_urls import url_decode, url_encode
from cudf._lib.strings.split.partition import partition, rpartition
from cudf._lib.strings.split.split import (
    rsplit,
    rsplit_re,
    rsplit_record,
    rsplit_record_re,
    split,
    split_re,
    split_record,
    split_record_re,
)
