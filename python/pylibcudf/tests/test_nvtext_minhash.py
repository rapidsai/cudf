# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.fixture(scope="module", params=[pa.uint32(), pa.uint64()])
def minhash_input_data(request):
    input_arr = pa.array(["foo", "bar", "foo foo", "bar bar"])
    seeds = pa.array([2, 3, 4, 5], request.param)
    return input_arr, seeds, request.param


@pytest.mark.parametrize("width", [5, 12])
def test_minhash(minhash_input_data, width):
    input_arr, seeds, seed_type = minhash_input_data
    minhash_func = (
        plc.nvtext.minhash.minhash
        if seed_type == pa.uint32()
        else plc.nvtext.minhash.minhash64
    )
    pa_result = minhash_func(
        plc.Column.from_arrow(input_arr),
        0,
        plc.Column.from_arrow(seeds),
        plc.Column.from_arrow(seeds),
        width,
    ).to_arrow()
    assert all(
        len(got) == len(seeds)
        for got, s in zip(pa_result, input_arr, strict=True)
    )
    assert pa_result.type == pa.list_(
        pa.field("element", seed_type, nullable=False)
    )


@pytest.fixture(scope="module", params=[pa.uint32(), pa.uint64()])
def minhash_ngrams_input_data(request):
    input_arr = pa.array(
        [
            ["foo", "bar", "foo foo", "bar bar", "foo bar", "bar foo"],
            [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
            ],
        ]
    )
    ab = pa.array([2, 3, 4, 5], request.param)
    return input_arr, ab, request.param


@pytest.mark.parametrize("ngrams", [5, 10])
def test_minhash_ngrams(minhash_ngrams_input_data, ngrams):
    input_arr, ab, seed_type = minhash_ngrams_input_data
    minhash_func = (
        plc.nvtext.minhash.minhash_ngrams
        if seed_type == pa.uint32()
        else plc.nvtext.minhash.minhash64_ngrams
    )
    pa_result = minhash_func(
        plc.Column.from_arrow(input_arr),
        ngrams,
        0,
        plc.Column.from_arrow(ab),
        plc.Column.from_arrow(ab),
    ).to_arrow()
    assert all(
        len(got) == len(ab)
        for got, s in zip(pa_result, input_arr, strict=True)
    )
    assert pa_result.type == pa.list_(
        pa.field("element", seed_type, nullable=False)
    )
