from __future__ import annotations
import cupy
from cudf._lib.nvtext.subword_tokenize import (
    subword_tokenize_inmem_hash as cpp_subword_tokenize,
    Hashed_Vocabulary as cpp_hashed_vocabulary
)

class subword_tokenizer:

    def __init__(self,
        hash_file: str,
        max_length: int = 64,
        stride: int = 48,
        do_lower: bool = True,
        do_truncate: bool = False,
        max_rows_tensor: int = 500,
    ):
        self.max_length = max_length
        self.stride = stride
        self.do_lower = do_lower
        self.do_truncate = do_truncate
        self.max_rows_tensor = max_rows_tensor
        self.vocab_file = cpp_hashed_vocabulary(hash_file) 

    
    def encode(self, str_series):
        tokens, masks, metadata = cpp_subword_tokenize(
            str_series._column,
            self.vocab_file,
            self.max_length,
            self.stride,
            self.do_lower,
            self.do_truncate,
            self.max_rows_tensor,
        )
        return (
            cupy.asarray(tokens).reshape(-1, self.max_length),
            cupy.asarray(masks).reshape(-1, self.max_length),
            cupy.asarray(metadata).reshape(-1, 3),
        ) 
