# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.fixture(scope="module", params=[pa.uint32(), pa.uint64()])
def minhash_input_data(request):
    input_arr = pa.array(["foo", "bar", "foo foo", "bar bar"])
    seeds = pa.array([2, 3, 4, 5], request.param)
    return input_arr, seeds, request.param


@pytest.fixture(scope="module", params=[pa.uint32(), pa.uint64()])
def word_minhash_input_data(request):
    input_arr = pa.array([["foo", "bar"], ["foo foo", "bar bar"]])
    seeds = pa.array([2, 3, 4, 5], request.param)
    return input_arr, seeds, request.param


@pytest.mark.parametrize("width", [5, 12])
def test_minhash_permuted(minhash_input_data, width):
    input_arr, seeds, seed_type = minhash_input_data
    minhash_func = (
        plc.nvtext.minhash.minhash_permuted
        if seed_type == pa.uint32()
        else plc.nvtext.minhash.minhash64_permuted
    )
    result = minhash_func(
        plc.interop.from_arrow(input_arr),
        0,
        plc.interop.from_arrow(seeds),
        plc.interop.from_arrow(seeds),
        width,
    )
    pa_result = plc.interop.to_arrow(result)
    assert all(len(got) == len(seeds) for got, s in zip(pa_result, input_arr))
    assert pa_result.type == pa.list_(
        pa.field("element", seed_type, nullable=False)
    )


def test_word_minhash(word_minhash_input_data):
    input_arr, seeds, seed_type = word_minhash_input_data
    word_minhash_func = (
        plc.nvtext.minhash.word_minhash
        if seed_type == pa.uint32()
        else plc.nvtext.minhash.word_minhash64
    )
    result = word_minhash_func(
        plc.interop.from_arrow(input_arr), plc.interop.from_arrow(seeds)
    )
    pa_result = plc.interop.to_arrow(result)
    assert all(len(got) == len(seeds) for got, s in zip(pa_result, input_arr))
    assert pa_result.type == pa.list_(
        pa.field("element", seed_type, nullable=False)
    )
