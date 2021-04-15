# Copyright (c) 2020-2021, NVIDIA CORPORATION.
from transformers import BertTokenizer
import pytest
import os


@pytest.fixture(scope="module")
def datadir(datadir):
    return os.path.join(datadir, "subword_tokenizer_data")


def assert_equal_tokenization_outputs(hf_output,cudf_output):
    assert np.sum(hf_output['input_ids']!=cudf_output['input_ids'].get())==0
    assert np.sum(hf_output['attention_mask']!=cudf_output['attention_mask'].get())==0

@pytest.mark.parametrize("seq_len", [32,64])
@pytest.mark.parametrize("stride", [0,15,32])
@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_subword_tokenize(seq_len,stride,add_special_tokens,datadir):
    with open(os.path.join(datadir,'test_sentences.txt')) as file:
        input_sentence_ls = [line.strip() for line in file]
    

    vocab_dir = 'data/bert_base_uncased_sampled'
    hf_tokenizer = BertTokenizerFast.from_pretrained(vocab_dir, do_lower_case)


    hf_output = hf_tokenizer(input_sentence_ls,
                             max_length=seq_len,
                             stride=stride,
                             padding='max_length',
                             return_tensors='np',
                             truncation=True,
                             add_special_tokens=add_special_tokens,
                            )

    
    str_series = cudf.Series(str_series)
    cudf_tokenizer  = SubwordTokenizer('bert_base_uncased_sampled/vocab-hash.txt', do_lower_case)
    cudf_output = cudf_tokenizer(str_series,
                     max_length=seq_len,
                     max_num_rows=seq_len,
                     stride=stride,
                     padding='max_length',
                     return_tensors='cp',
                     truncation=True,
                     add_special_tokens=add_special_tokens
                      )

    assert_equal_tokenization_outputs(cudf_output, hf_output)


    