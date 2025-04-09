# Copyright (c) 2020-2025, NVIDIA CORPORATION.
import os

import cupy
import numpy as np
import pytest

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer
from cudf.core.wordpiece_tokenize import WordPieceVocabulary
from cudf.testing import assert_eq


@pytest.fixture(scope="module")
def datadir(datadir):
    return os.path.join(datadir, "subword_tokenizer_data")


def assert_equal_tokenization_outputs(hf_output, cudf_output):
    assert (
        np.sum(hf_output["input_ids"] != cudf_output["input_ids"].get()) == 0
    )
    assert (
        np.sum(
            hf_output["attention_mask"] != cudf_output["attention_mask"].get()
        )
        == 0
    )


@pytest.mark.skip(reason="segfaults")
@pytest.mark.parametrize("seq_len", [32, 64])
@pytest.mark.parametrize("stride", [0, 15, 30])
@pytest.mark.parametrize("add_special_tokens", [True, False])
@pytest.mark.parametrize("do_lower_case", [True, False])
def test_subword_tokenize(
    seq_len, stride, add_special_tokens, do_lower_case, datadir
):
    with open(
        os.path.join(datadir, "test_sentences.txt"), encoding="utf-8"
    ) as file:
        input_sentence_ls = [line.strip() for line in file]

    vocab_dir = os.path.join(datadir, "bert_base_cased_sampled")

    transformers = pytest.importorskip("transformers")

    hf_tokenizer = transformers.BertTokenizer.from_pretrained(
        vocab_dir, do_lower_case=do_lower_case
    )

    hf_output = hf_tokenizer(
        input_sentence_ls,
        max_length=seq_len,
        stride=stride,
        padding="max_length",
        return_tensors="np",
        truncation=True,
        add_special_tokens=add_special_tokens,
    )

    vocab_hash = os.path.join(vocab_dir, "vocab-hash.txt")
    str_series = cudf.Series(input_sentence_ls)
    cudf_tokenizer = SubwordTokenizer(vocab_hash, do_lower_case=do_lower_case)
    cudf_output = cudf_tokenizer(
        str_series,
        max_length=seq_len,
        max_num_rows=len(str_series),
        stride=stride,
        padding="max_length",
        return_tensors="cp",
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    assert_equal_tokenization_outputs(hf_output, cudf_output)


def test_subword_tokenize_with_truncation(datadir):
    vocab_dir = os.path.join(datadir, "bert_base_cased_sampled")
    vocab_hash = os.path.join(vocab_dir, "vocab-hash.txt")
    str_series = cudf.Series(["Test error"])
    cudf_tokenizer = SubwordTokenizer(vocab_hash)

    error_msg = (
        "Adding special tokens is not supported with truncation = False. "
        "Custom Cupy kernel can potentially "
        "be used to add it. For reference "
        "see: _bert_add_special_tokens"
    )

    with pytest.raises(NotImplementedError, match=error_msg):
        cudf_tokenizer(
            str_series,
            max_length=64,
            max_num_rows=len(str_series),
            truncation=False,
            add_special_tokens=True,
        )


def test_text_subword_tokenize(tmpdir):
    sr = cudf.Series(
        [
            "This is a test",
            "A test this is",
            "Is test a this",
            "Test   test",
            "this   This",
        ]
    )
    hash_file = tmpdir.mkdir("nvtext").join("tmp_hashed_vocab.txt")
    content = "1\n0\n23\n"
    coefficients = [65559] * 23
    for c in coefficients:
        content = content + str(c) + " 0\n"
    # based on values from the bert_hash_table.txt file for the
    # test words used here: 'this' 'is' 'a' test'
    table = [0] * 23
    table[0] = 3015668
    table[1] = 6205475701751155871
    table[5] = 6358029
    table[16] = 451412625363
    table[20] = 6206321707968235495
    content = content + "23\n"
    for v in table:
        content = content + str(v) + "\n"
    content = content + "100\n101\n102\n\n"
    hash_file.write(content)

    cudf_tokenizer = SubwordTokenizer(hash_file)

    token_d = cudf_tokenizer(
        sr, 8, 8, add_special_tokens=False, truncation=True
    )
    tokens, masks, metadata = (
        token_d["input_ids"],
        token_d["attention_mask"],
        token_d["metadata"],
    )
    expected_tokens = cupy.asarray(
        [
            2023,
            2003,
            1037,
            3231,
            0,
            0,
            0,
            0,
            1037,
            3231,
            2023,
            2003,
            0,
            0,
            0,
            0,
            2003,
            3231,
            1037,
            2023,
            0,
            0,
            0,
            0,
            3231,
            3231,
            0,
            0,
            0,
            0,
            0,
            0,
            2023,
            2023,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.uint32,
    )
    expected_tokens = expected_tokens.reshape(-1, 8)
    assert_eq(expected_tokens, tokens)

    expected_masks = cupy.asarray(
        [
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.uint32,
    )
    expected_masks = expected_masks.reshape(-1, 8)
    assert_eq(expected_masks, masks)

    expected_metadata = cupy.asarray(
        [0, 0, 3, 1, 0, 3, 2, 0, 3, 3, 0, 1, 4, 0, 1], dtype=np.uint32
    )
    expected_metadata = expected_metadata.reshape(-1, 3)
    assert_eq(expected_metadata, metadata)


@pytest.mark.parametrize("max_words", [0, 200, 10])
def test_text_wordpiece_tokenize(max_words, datadir):
    s = cudf.Series(
        [
            "The British Isles have been ringing for the last few years with the word  ' Art '  in its German sense ; ",
            "with  ' High Art ,  '   ' Symbolic Art ,  '   ' Ecclesiastical Art ,  '   ' Dramatic Art ,  '   ' Tragic Art ,  '  and so forth ; ",
            "and every well - educated person is expected ,  nowadays ,  to know something about Art . "
            "Yet in spite of all translations of German  ' AEsthetic '  treatises ,  and  ' Kunstnovellen ,  '  the mass of the British people cares very little about the matter ,  and sits contented under the imputation of  ' bad taste . ",
            " ' Our stage ,  long since dead ,  does not revive ;  our poetry is dying ;  our music ,  like our architecture ,  only reproduces the past ; ",
            "our painting is only first - rate when it handles landscapes and animals ,  and seems likely so to remain ; ",
            "but ,  meanwhile ,  nobody cares .   Some of the deepest and most earnest minds vote the question ,  in general ,  a  ' sham and a snare ,  '  and whisper to each other",
        ]
    )
    vocab_file = os.path.join(datadir, "bert_base_cased_sampled/vocab.txt")
    vc = cudf.read_text(vocab_file, delimiter="\n", strip_delimiters=True)
    wpt = WordPieceVocabulary(vc)
    wpr = wpt.tokenize(s, max_words)
    expected = cudf.Series(
        [
            cudf.Series(
                [
                    1109,
                    1418,
                    2181,
                    2897,
                    1138,
                    1151,
                    3170,
                    1158,
                    1111,
                    1103,
                    1314,
                    1374,
                    1201,
                    1114,
                    1103,
                    1937,
                    112,
                    2051,
                    112,
                    1107,
                    1157,
                    1528,
                    2305,
                    132,
                ],
                dtype=np.int32,
            ),
            cudf.Series(
                [
                    1114,
                    112,
                    1693,
                    2051,
                    117,
                    112,
                    112,
                    156,
                    1183,
                    1306,
                    1830,
                    1186,
                    2646,
                    1665,
                    2051,
                    117,
                    112,
                    112,
                    142,
                    1665,
                    1665,
                    2897,
                    1465,
                    2050,
                    1596,
                    1348,
                    2051,
                    117,
                    112,
                    112,
                    1987,
                    2312,
                    2980,
                    1596,
                    2051,
                    117,
                    112,
                    112,
                    157,
                    1611,
                    1403,
                    1596,
                    2051,
                    117,
                    112,
                    1105,
                    1177,
                    1111,
                    1582,
                    132,
                ],
                dtype=np.int32,
            ),
            cudf.Series(
                [
                    1105,
                    1451,
                    1218,
                    118,
                    174,
                    1181,
                    1358,
                    2599,
                    1906,
                    1825,
                    1110,
                    2637,
                    117,
                    1208,
                    1161,
                    1810,
                    1183,
                    1116,
                    117,
                    1106,
                    1221,
                    1380,
                    1164,
                    2051,
                    119,
                    162,
                    2105,
                    1107,
                    188,
                    1643,
                    3150,
                    1104,
                    1155,
                    189,
                    1611,
                    2316,
                    1742,
                    2116,
                    1116,
                    1104,
                    1528,
                    112,
                    138,
                    2036,
                    2050,
                    1324,
                    2105,
                    1596,
                    112,
                    189,
                    1874,
                    2980,
                    1548,
                    1279,
                    117,
                    1105,
                    112,
                    148,
                    3488,
                    2050,
                    2728,
                    2707,
                    2339,
                    1424,
                    117,
                    112,
                    1103,
                    3367,
                    1104,
                    1103,
                    1418,
                    1234,
                    1920,
                    1116,
                    1304,
                    1376,
                    1164,
                    1103,
                    2187,
                    117,
                    1105,
                    3465,
                    1116,
                    3438,
                    1174,
                    1223,
                    1103,
                    178,
                    1306,
                    1643,
                    1358,
                    1777,
                    2116,
                    1104,
                    112,
                    2213,
                    189,
                    2225,
                    1566,
                    119,
                ],
                dtype=np.int32,
            ),
            cudf.Series(
                [
                    112,
                    3458,
                    2016,
                    117,
                    1263,
                    1290,
                    2044,
                    117,
                    1674,
                    1136,
                    1231,
                    1964,
                    2109,
                    132,
                    1412,
                    185,
                    1186,
                    2105,
                    1616,
                    1110,
                    173,
                    1183,
                    1158,
                    132,
                    1412,
                    1390,
                    117,
                    1176,
                    1412,
                    170,
                    1197,
                    1732,
                    3150,
                    1665,
                    1204,
                    3313,
                    117,
                    1178,
                    1231,
                    1643,
                    2180,
                    1181,
                    1358,
                    2093,
                    1116,
                    1103,
                    1763,
                    132,
                ],
                dtype=np.int32,
            ),
            cudf.Series(
                [
                    1412,
                    2489,
                    1916,
                    1110,
                    1178,
                    1148,
                    118,
                    2603,
                    1165,
                    1122,
                    1289,
                    2897,
                    1657,
                    1116,
                    2599,
                    3186,
                    1116,
                    1105,
                    1126,
                    1182,
                    1918,
                    3447,
                    117,
                    1105,
                    3093,
                    2620,
                    1177,
                    1106,
                    3118,
                    132,
                ],
                dtype=np.int32,
            ),
            cudf.Series(
                [
                    1133,
                    117,
                    1928,
                    2246,
                    3031,
                    1513,
                    117,
                    1185,
                    1830,
                    1186,
                    1181,
                    1183,
                    1920,
                    1116,
                    119,
                    1789,
                    1104,
                    1103,
                    1996,
                    2556,
                    1105,
                    1211,
                    174,
                    1813,
                    1673,
                    2050,
                    1713,
                    1116,
                    2992,
                    1103,
                    2304,
                    117,
                    1107,
                    1704,
                    117,
                    170,
                    112,
                    188,
                    2522,
                    1105,
                    170,
                    188,
                    1605,
                    1874,
                    117,
                    112,
                    1105,
                    192,
                    3031,
                    1116,
                    3365,
                    1106,
                    1296,
                    1168,
                ],
                dtype=np.int32,
            ),
        ]
    )
    if max_words == 10:
        expected = cudf.Series(
            [
                cudf.Series(
                    [
                        1109,
                        1418,
                        2181,
                        2897,
                        1138,
                        1151,
                        3170,
                        1158,
                        1111,
                        1103,
                        1314,
                        1374,
                    ],
                    dtype=np.int32,
                ),
                cudf.Series(
                    [
                        1114,
                        112,
                        1693,
                        2051,
                        117,
                        112,
                        112,
                        156,
                        1183,
                        1306,
                        1830,
                        1186,
                        2646,
                        1665,
                        2051,
                        117,
                    ],
                    dtype=np.int32,
                ),
                cudf.Series(
                    [
                        1105,
                        1451,
                        1218,
                        118,
                        174,
                        1181,
                        1358,
                        2599,
                        1906,
                        1825,
                        1110,
                        2637,
                        117,
                        1208,
                        1161,
                        1810,
                        1183,
                        1116,
                    ],
                    dtype=np.int32,
                ),
                cudf.Series(
                    [112, 3458, 2016, 117, 1263, 1290, 2044, 117, 1674, 1136],
                    dtype=np.int32,
                ),
                cudf.Series(
                    [
                        1412,
                        2489,
                        1916,
                        1110,
                        1178,
                        1148,
                        118,
                        2603,
                        1165,
                        1122,
                        1289,
                        2897,
                    ],
                    dtype=np.int32,
                ),
                cudf.Series(
                    [
                        1133,
                        117,
                        1928,
                        2246,
                        3031,
                        1513,
                        117,
                        1185,
                        1830,
                        1186,
                        1181,
                        1183,
                        1920,
                        1116,
                        119,
                        1789,
                        1104,
                        1103,
                    ],
                    dtype=np.int32,
                ),
            ]
        )
    assert_eq(expected, wpr)
