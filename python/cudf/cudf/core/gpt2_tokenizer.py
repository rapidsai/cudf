from cudf.core.byte_pair_encoding import BytePairEncoder
from cudf.core.tokenize_vocabulary import TokenizeVocabulary
import cudf
from typing import Dict
from cudf.core.column import as_column, ListColumn
import cupy as cp

class GPT2Tokenizer:
    """
    Run CUDA GPT2 subword tokenizer on cuDF strings column.
    Encodes words to token ids using vocabulary from a pretrained
    tokenizer.

    TODO : In future this could accept merge file and a vocab file, and maybe allow a from_pretrained method
    Parameters
    ----------
    vocab : cudf.Series
        List of words in vocabulary ordered by token id
    bpe_merges : cudf.Series
        Series containing the Byte Pair merges
        Default `https://huggingface.co/gpt2/raw/main/merges.txt`
    Returns
    -------
    GPT2Tokenizer
    """

    def __init__(self, vocab: cudf.Series, bpe_merges: cudf.Series):
        self.encode_to_token_ids = TokenizeVocabulary(vocab)
        self.bpe = BytePairEncoder(bpe_merges)

        # The original regex is from here
        # github.com/openai/tiktoken/blob/1b9faf2779855124f05174adf1383e53689ed94b/tiktoken_ext/openai_public.py#L23C20-L23C98
        # There are two broad issues here:
        # 1. Unicode pattern matching of \p{L} and \p{N}, replaced here with \w and \d
        # 2. Per github.com/rapidsai/cudf/issues/3100 negative lookahead are not supported raises question
        #       how we can support `\s+(?!\S)`, for the POC have removed it
        self.pat =  r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+"""

        self.byte_encoder = self._bytes_to_unicode()

    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        """
        Cloned from https://github.com/huggingface/transformers/blob/c5f0288bc7d76f65996586f79f69fba8867a0e67/src/transformers/models/gpt2/tokenization_gpt2.py#L62-L84
        Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
        characters the bpe code barfs on.

        The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
        if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
        decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
        tables between utf-8 bytes and unicode strings.
        """

        numeric_encoding = (
                list(range(ord("!"), ord("~") + 1)) +
                list(range(ord("¡"), ord("¬") + 1)) +
                list(range(ord("®"), ord("ÿ") + 1))
        )
        character_encoding = numeric_encoding[:]


        counter = 0

        for i in range(2 ** 8):
            if i not in numeric_encoding:
                numeric_encoding.append(i)
                # Add a corresponding value to the character list (256 + counter)
                character_encoding.append(2 ** 8 + counter)
                counter += 1
        # Convert list into actual characters
        character_encoding = [chr(n) for n in character_encoding]
        return dict(zip(numeric_encoding, character_encoding))


    def _slice(self, data : cudf.Series, start_offsets: cudf.Series) -> cudf.Series:
        """
        Given an input cudf.Series where each element is of List[Any]
         and start_offsets, return a new cudf.Series where each element is
            a List[Any] obtained by slicing the corresponding element in `data`
        """

        assert start_offsets.iloc[-1] == len(data), f"Last element of `start_offsets` should be the same as {len(data)=}"
        n_rows = len(start_offsets) - 1
        # Create a column of start_offsets (important to cast to int32)
        offset_col = as_column(start_offsets, dtype="int32")

        mask_col = cp.full(shape=n_rows, fill_value=True)
        mask = cudf._lib.transform.bools_to_mask(as_column(mask_col))
        output_col = ListColumn(
            data=None,
            size=n_rows,
            dtype=cudf.ListDtype(data.dtype),
            mask=mask,
            offset=0,
            null_count=0,
            children=(offset_col, as_column(data)),
        )

        return cudf.Series._from_column(output_col).list.concat()

    def convert_to_encoded_str_tokens(self, text: cudf.Series) -> cudf.Series:
        """
        Convert a cudf.Series of strings to a cudf.Series of encoded tokens
        """
        # Convert Series(str) to Series(List[str])
        # eg "This is a test" -> ["This", " is", " a", " test"]
        tokens = text.str.findall(self.pat)

        # Flatten Series(List[str]) to Series(str)
        # eg [["This", " is", " a", " test"], ["This", " is", " another"]] becomes
        #    ["This", " is", " a", " test", "This", " is", " another"]
        flattened_token = tokens.list.leaves

        # ["This", " is", " a", " test"] -> [b"This", b" is", b" a", b" test"]
        # TODO This fails on strings with non-ascii characters, and rather produces incorrect results
        # We need a way to encode the string in utf-8, so that our translate can be provided with only bytes
        # Currently the translate expects unicode points
        # Eg if we encode `ಠ` to utf-8, we get b'\xe0\xb2\xa0'
        # We want our translate to work on b'\xe0\xb2\xa0' and not on `ಠ`
        flattened_tokens_in_encoded_bytes = flattened_token.str.translate(self.byte_encoder)

        # Series(str) -> Series(str)
        # Run BPE on the encoded tokens
        bpe_tokens = self.bpe(flattened_tokens_in_encoded_bytes)

        # Series(str) -> Series(List[List[int]])
        # For each of the BPE Tokens, convert them to token_ids now
        # eg ["This", " is", " a", " test"] -> [[1], [2], [3], [4]]
        token_ids = self.encode_to_token_ids.tokenize(bpe_tokens)

        # len(token_ids) is the same as len(flattened_tokens)
        # We need to convert it back to the original shape i.e len(text)
        # eg ["This", " is", " a", " test", "This", " is", " another"] -> [[1, 2, 3, 4], [1, 2, 5]]
        # We also need to go from Series(List[List[int]]) to Series(List[int])

        cudf_tokens_start = tokens.list.len().cumsum() - tokens.list.len()
        start_indices = cudf.concat([cudf_tokens_start, cudf.Series([len(flattened_token)])])
        return self._slice(token_ids, start_indices)



    def __call__(
        self,
        text : cudf.Series,
    ) -> cudf.Series:
        """ Tokenize a cudf.Series of strings to a cudf.Series of token ids

        TODO : Add functionality similar to pad / truncate
        TODO : Make output match HuggingFace output i.e {'input_ids': cudf.Series, 'attention_mask': cudf.Series}
        TODO : Allow option to return pytorch / tensorflow tensors instead of cudf.Series
        """
        return self.convert_to_encoded_str_tokens(text)
