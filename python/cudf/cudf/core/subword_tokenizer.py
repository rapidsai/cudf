from __future__ import annotations
import cupy
from cudf._lib.nvtext.subword_tokenize import (
    subword_tokenize_inmem_hash as cpp_subword_tokenize,
    Hashed_Vocabulary as cpp_hashed_vocabulary
)

class subword_tokenizer:

    def __init__(self,
        hash_file: str,
        do_lower: bool = True,
        do_truncate: bool = False,
    ):
        self.do_lower = do_lower
        self.do_truncate = do_truncate
        self.vocab_file = cpp_hashed_vocabulary(hash_file) 

    
    def encode(self, str_series, max_length,stride,max_rows_tensor):
        tokens, masks, metadata = cpp_subword_tokenize(
            str_series._column,
            self.vocab_file,
            max_length,
            stride,
            self.do_lower,
            self.do_truncate,
            max_rows_tensor,
        )
        return (
            cupy.asarray(tokens).reshape(-1, max_length),
            cupy.asarray(masks).reshape(-1, max_length),
            cupy.asarray(metadata).reshape(-1, 3),
        ) 
