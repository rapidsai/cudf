# Copyright (c) 2018-2024, NVIDIA CORPORATION.

"""
Test method that apply GPU kernel to a frame.
"""

import numpy as np
import pytest
from numba import cuda

from cudf import DataFrame
from cudf.testing import assert_eq


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

    got_out1 = outdf["out1"].to_numpy()
    got_out2 = outdf["out2"].to_numpy()

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

    np.testing.assert_array_almost_equal(got_out1.to_numpy(), expect_out1)
    np.testing.assert_array_almost_equal(got_out2.to_numpy(), expect_out2)


@pytest.mark.parametrize("nelem", [1, 15, 30, 64, 128, 129])
def test_df_apply_custom_chunks(nelem):
    def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        for i, (x, y, z) in enumerate(zip(in1, in2, in3)):
            out1[i] = extra2 * x - extra1 * y + z
            out2[i] = i

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
        [np.arange(e - s) for s, e in zip(chunks, chunks[1:] + [len(df)])]
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

    np.testing.assert_array_almost_equal(got_out1.to_numpy(), expect_out1)
    np.testing.assert_array_almost_equal(got_out2.to_numpy(), expect_out2)


@pytest.mark.parametrize("nelem", [1, 15, 30, 64, 128, 129])
@pytest.mark.parametrize("blkct", [None, 1, 8])
@pytest.mark.parametrize("tpb", [1, 8, 64])
def test_df_apply_custom_chunks_blkct_tpb(nelem, blkct, tpb):
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
        [
            tpb * np.arange(e - s)
            for s, e in zip(chunks, chunks[1:] + [len(df)])
        ]
    )

    outdf = df.apply_chunks(
        kernel,
        incols=["in1", "in2", "in3"],
        outcols=dict(out1=np.float64, out2=np.int32),
        kwargs=dict(extra1=extra1, extra2=extra2),
        chunks=chunks,
        blkct=blkct,
        tpb=tpb,
    )

    got_out1 = outdf["out1"]
    got_out2 = outdf["out2"]

    np.testing.assert_array_almost_equal(got_out1.to_numpy(), expect_out1)
    np.testing.assert_array_almost_equal(got_out2.to_numpy(), expect_out2)


@pytest.mark.parametrize("nelem", [1, 2, 64, 128, 1000, 5000])
def test_df_apply_rows_incols_mapping(nelem):
    def kernel(x, y, z, out1, out2, extra1, extra2):
        for i, (a, b, c) in enumerate(zip(x, y, z)):
            out1[i] = extra2 * a - extra1 * b
            out2[i] = b - extra1 * c

    df = DataFrame()
    df["in1"] = in1 = np.arange(nelem)
    df["in2"] = in2 = np.arange(nelem)
    df["in3"] = in3 = np.arange(nelem)

    extra1 = 2.3
    extra2 = 3.4

    expected_out = DataFrame()
    expected_out["out1"] = extra2 * in1 - extra1 * in2
    expected_out["out2"] = in2 - extra1 * in3

    outdf = df.apply_rows(
        kernel,
        incols={"in1": "x", "in2": "y", "in3": "z"},
        outcols=dict(out1=np.float64, out2=np.float64),
        kwargs=dict(extra1=extra1, extra2=extra2),
    )

    assert_eq(outdf[["out1", "out2"]], expected_out)


@pytest.mark.parametrize("nelem", [1, 2, 64, 128, 129])
@pytest.mark.parametrize("chunksize", [1, 2, 3, 4, 23])
def test_df_apply_chunks_incols_mapping(nelem, chunksize):
    def kernel(q, p, r, out1, out2, extra1, extra2):
        for i, (a, b, c) in enumerate(zip(q, p, r)):
            out1[i] = extra2 * a - extra1 * b + c
            out2[i] = i

    df = DataFrame()
    df["in1"] = in1 = np.arange(nelem)
    df["in2"] = in2 = np.arange(nelem)
    df["in3"] = in3 = np.arange(nelem)

    extra1 = 2.3
    extra2 = 3.4

    expected_out = DataFrame()
    expected_out["out1"] = extra2 * in1 - extra1 * in2 + in3
    expected_out["out2"] = np.arange(len(df)) % chunksize

    outdf = df.apply_chunks(
        kernel,
        incols={"in1": "q", "in2": "p", "in3": "r"},
        outcols=dict(out1=np.float64, out2=np.int64),
        kwargs=dict(extra1=extra1, extra2=extra2),
        chunks=chunksize,
    )

    assert_eq(outdf[["out1", "out2"]], expected_out)
