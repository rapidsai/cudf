

from transformers import BertTokenizer


def assert_equal_tokenization_outputs(hf_output,cudf_output):
    assert np.sum(hf_output['input_ids']!=cudf_output['input_ids'].get())==0
    assert np.sum(hf_output['attention_mask']!=cudf_output['attention_mask'].get())==0


def test_subword_tokenize():
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


    