# Copyright (c) 2018, NVIDIA CORPORATION.

"""
Test method that apply GPU kernel to a frame.
"""

import numpy as np
import pytest
from numba import cuda

from cudf import DataFrame


@pytest.mark.parametrize("nelem", [1, 2, 64, 128, 129])
def test_df_apply_rows(nelem):
    def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        for i, (x, y, z) in enumerate(zip(in1, in2, in3)):
            out1[i] = extra2 * x - extra1 * y
            out2[i] = y - extra1 * z

    df = DataFrame()
    df["in1"] = in1 = np.arange(nelem)
    df["in2"] = in2 = np.arange(nelem)
    df["in3"] = in3 = np.arange(nelem)

    extra1 = 2.3
    extra2 = 3.4

    expect_out1 = extra2 * in1 - extra1 * in2
    expect_out2 = in2 - extra1 * in3

    outdf = df.apply_rows(
        kernel,
        incols=["in1", "in2", "in3"],
        outcols=dict(out1=np.float64, out2=np.float64),
        kwargs=dict(extra1=extra1, extra2=extra2),
    )

    got_out1 = outdf["out1"].to_array()
    got_out2 = outdf["out2"].to_array()

    np.testing.assert_array_almost_equal(got_out1, expect_out1)
    np.testing.assert_array_almost_equal(got_out2, expect_out2)


@pytest.mark.parametrize("nelem", [1, 2, 64, 128, 129])
@pytest.mark.parametrize("chunksize", [1, 2, 3, 4, 23])
def test_df_apply_chunks(nelem, chunksize):
    def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        for i, (x, y, z) in enumerate(zip(in1, in2, in3)):
            out1[i] = extra2 * x - extra1 * y + z
            out2[i] = i

    df = DataFrame()
    df["in1"] = in1 = np.arange(nelem)
    df["in2"] = in2 = np.arange(nelem)
    df["in3"] = in3 = np.arange(nelem)

    extra1 = 2.3
    extra2 = 3.4

    expect_out1 = extra2 * in1 - extra1 * in2 + in3
    expect_out2 = np.arange(len(df)) % chunksize

    outdf = df.apply_chunks(
        kernel,
        incols=["in1", "in2", "in3"],
        outcols=dict(out1=np.float64, out2=np.int32),
        kwargs=dict(extra1=extra1, extra2=extra2),
        chunks=chunksize,
    )

    got_out1 = outdf["out1"]
    got_out2 = outdf["out2"]

    np.testing.assert_array_almost_equal(got_out1, expect_out1)
    np.testing.assert_array_almost_equal(got_out2, expect_out2)


@pytest.mark.parametrize("nelem", [1, 15, 30, 64, 128, 129])
def test_df_apply_custom_chunks(nelem):
    def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        for i, (x, y, z) in enumerate(zip(in1, in2, in3)):
            out1[i] = extra2 * x - extra1 * y + z
            # cuda.blockDim.x is 1
            out2[i] = i * cuda.blockDim.x

    df = DataFrame()
    df["in1"] = in1 = np.arange(nelem)
    df["in2"] = in2 = np.arange(nelem)
    df["in3"] = in3 = np.arange(nelem)

    chunks = [0, 7, 11, 29, 101, 777]
    chunks = [c for c in chunks if c < nelem]

    extra1 = 2.3
    extra2 = 3.4

    expect_out1 = extra2 * in1 - extra1 * in2 + in3
    expect_out2 = np.hstack(
        np.arange((e - s)) for s, e in zip(chunks, chunks[1:] + [len(df)])
    )

    outdf = df.apply_chunks(
        kernel,
        incols=["in1", "in2", "in3"],
        outcols=dict(out1=np.float64, out2=np.int32),
        kwargs=dict(extra1=extra1, extra2=extra2),
        chunks=chunks,
    )

    got_out1 = outdf["out1"]
    got_out2 = outdf["out2"]

    np.testing.assert_array_almost_equal(got_out1, expect_out1)
    np.testing.assert_array_almost_equal(got_out2, expect_out2)


@pytest.mark.parametrize("nelem", [1, 15, 30, 64, 128, 129])
@pytest.mark.parametrize("tpb", [1, 8, 16, 64])
def test_df_apply_custom_chunks_tpb(nelem, tpb):
    def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        for i in range(cuda.threadIdx.x, in1.size, cuda.blockDim.x):
            x = in1[i]
            y = in2[i]
            z = in3[i]
            out1[i] = extra2 * x - extra1 * y + z
            out2[i] = i * cuda.blockDim.x

    df = DataFrame()
    df["in1"] = in1 = np.arange(nelem)
    df["in2"] = in2 = np.arange(nelem)
    df["in3"] = in3 = np.arange(nelem)

    chunks = [0, 7, 11, 29, 101, 777]
    chunks = [c for c in chunks if c < nelem]

    extra1 = 2.3
    extra2 = 3.4

    expect_out1 = extra2 * in1 - extra1 * in2 + in3
    expect_out2 = np.hstack(
        tpb * np.arange((e - s))
        for s, e in zip(chunks, chunks[1:] + [len(df)])
    )

    outdf = df.apply_chunks(
        kernel,
        incols=["in1", "in2", "in3"],
        outcols=dict(out1=np.float64, out2=np.int32),
        kwargs=dict(extra1=extra1, extra2=extra2),
        chunks=chunks,
        tpb=tpb,
    )

    got_out1 = outdf["out1"]
    got_out2 = outdf["out2"]

    np.testing.assert_array_almost_equal(got_out1, expect_out1)
    np.testing.assert_array_almost_equal(got_out2, expect_out2)
