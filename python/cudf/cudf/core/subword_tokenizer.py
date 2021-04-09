from __future__ import annotations
from cudf._lib.nvtext.subword_tokenize import (
    Hashed_Vocabulary as c_hashed_vocabulary,
)
import cupy
from cudf._lib.nvtext.subword_tokenize import (
    subword_tokenize as cpp_subword_tokenize,
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
        self.vocab_file = c_hashed_vocabulary(hash_file) 

    
    def encode(self, str_series):
        tokens, masks, metadata = cpp_subword_tokenize(
            self._column,
            self.vocab_file,
            self.max_length,
            self.stride,
            self.do_lower,
            self.do_truncate,
            self.max_rows_tensor,
        )
        return (
            cupy.asarray(tokens),
            cupy.asarray(masks),
            cupy.asarray(metadata),
        ) 
