String handling
~~~~~~~~~~~~~~~

``Series.str`` can be used to access the values of the series as
strings and apply several methods to it. These can be accessed like
``Series.str.<function/property>``.

.. currentmodule:: cudf
.. autosummary::
   :toctree: api/

   Series.str

.. currentmodule:: cudf.core.column.string.StringMethods
.. autosummary::
   :toctree: api/

   byte_count
   capitalize
   cat
   center
   character_ngrams
   character_tokenize
   code_points
   contains
   count
   detokenize
   edit_distance
   edit_distance_matrix
   endswith
   extract
   filter_alphanum
   filter_characters
   filter_tokens
   find
   findall
   find_multiple
   get
   get_json_object
   hex_to_int
   htoi
   index
   insert
   ip2int
   ip_to_int
   is_consonant
   is_vowel
   isalnum
   isalpha
   isdecimal
   isdigit
   isempty
   isfloat
   ishex
   isinteger
   isipv4
   isspace
   islower
   isnumeric
   isupper
   istimestamp
   istitle
   jaccard_index
   join
   len
   like
   ljust
   lower
   lstrip
   match
   minhash
   ngrams
   ngrams_tokenize
   normalize_characters
   normalize_spaces
   pad
   partition
   porter_stemmer_measure
   repeat
   removeprefix
   removesuffix
   replace
   replace_tokens
   replace_with_backrefs
   rfind
   rindex
   rjust
   rpartition
   rsplit
   rstrip
   slice
   slice_from
   slice_replace
   split
   startswith
   strip
   swapcase
   title
   token_count
   tokenize
   translate
   upper
   url_decode
   url_encode
   wrap
   zfill
