# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pytest
from numba import cuda

from cudf import DataFrame
from cudf.testing import assert_eq


@pytest.mark.parametrize("chunksize", [1, 4, 23])
def test_df_apply_chunks(chunksize):
    nelem = 20

    def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        for i, (x, y, z) in enumerate(zip(in1, in2, in3)):  # noqa: B905
            out1[i] = extra2 * x - extra1 * y + z
            out2[i] = i

    in1 = np.arange(nelem)
    in2 = np.arange(nelem)
    in3 = np.arange(nelem)

    df = DataFrame(
        {
            "in1": in1,
            "in2": in2,
            "in3": in3,
        }
    )

    extra1 = 2.3
    extra2 = 3.4

    expect_out1 = extra2 * in1 - extra1 * in2 + in3
    expect_out2 = np.arange(len(df)) % chunksize
    with pytest.warns(FutureWarning):
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


def test_df_apply_custom_chunks():
    nelem = 20

    def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        for i, (x, y, z) in enumerate(zip(in1, in2, in3)):  # noqa: B905
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
        [
            np.arange(e - s)
            for s, e in zip(chunks, chunks[1:] + [len(df)], strict=True)
        ]
    )

    with pytest.warns(FutureWarning):
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


@pytest.mark.parametrize("blkct", [None, 1, 8])
@pytest.mark.parametrize("tpb", [1, 8])
def test_df_apply_custom_chunks_blkct_tpb(blkct, tpb):
    nelem = 20

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
            for s, e in zip(chunks, chunks[1:] + [len(df)], strict=True)
        ]
    )

    with pytest.warns(FutureWarning):
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


@pytest.mark.parametrize("chunksize", [1, 4, 23])
def test_df_apply_chunks_incols_mapping(chunksize):
    nelem = 20

    def kernel(q, p, r, out1, out2, extra1, extra2):
        for i, (a, b, c) in enumerate(zip(q, p, r)):  # noqa: B905
            out1[i] = extra2 * a - extra1 * b + c
            out2[i] = i

    in1 = np.arange(nelem)
    in2 = np.arange(nelem)
    in3 = np.arange(nelem)

    df = DataFrame(
        {
            "in1": in1,
            "in2": in2,
            "in3": in3,
        }
    )

    extra1 = 2.3
    extra2 = 3.4

    expected_out = DataFrame(
        {
            "out1": extra2 * in1 - extra1 * in2 + in3,
            "out2": np.arange(len(df)) % chunksize,
        }
    )
    with pytest.warns(FutureWarning):
        outdf = df.apply_chunks(
            kernel,
            incols={"in1": "q", "in2": "p", "in3": "r"},
            outcols=dict(out1=np.float64, out2=np.int64),
            kwargs=dict(extra1=extra1, extra2=extra2),
            chunks=chunksize,
        )

    assert_eq(outdf[["out1", "out2"]], expected_out)
