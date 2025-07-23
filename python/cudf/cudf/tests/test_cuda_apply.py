# Copyright (c) 2018-2025, NVIDIA CORPORATION.

"""
Test method that apply GPU kernel to a frame.
"""

import numpy as np
import pytest
from numba import cuda

import cudf
from cudf import DataFrame
from cudf.core.column import column
from cudf.testing import assert_eq
from cudf.testing._utils import gen_rand_series


def _kernel_multiply(a, b, out):
    for i, (x, y) in enumerate(zip(a, b)):
        out[i] = x * y


@pytest.mark.parametrize("dtype", [np.dtype("float32"), np.dtype("float64")])
@pytest.mark.parametrize("has_nulls", [False, True])
@pytest.mark.parametrize("pessimistic", [False, True])
def test_dataframe_apply_rows(dtype, has_nulls, pessimistic):
    count = 1000
    gdf_series_a = gen_rand_series(dtype, count, has_nulls=has_nulls)
    gdf_series_b = gen_rand_series(dtype, count, has_nulls=has_nulls)
    gdf_series_c = gen_rand_series(dtype, count, has_nulls=has_nulls)

    if pessimistic:
        # pessimistically combine the null masks
        gdf_series_expected = gdf_series_a * gdf_series_b
    else:
        # optimistically ignore the null masks
        a = cudf.Series._from_column(
            column.build_column(gdf_series_a.data, dtype)
        )
        b = cudf.Series._from_column(
            column.build_column(gdf_series_b.data, dtype)
        )
        gdf_series_expected = a * b

    df_expected = cudf.DataFrame(
        {
            "a": gdf_series_a,
            "b": gdf_series_b,
            "c": gdf_series_c,
            "out": gdf_series_expected,
        }
    )

    df_original = cudf.DataFrame(
        {"a": gdf_series_a, "b": gdf_series_b, "c": gdf_series_c}
    )

    with pytest.warns(FutureWarning):
        df_actual = df_original.apply_rows(
            _kernel_multiply,
            ["a", "b"],
            {"out": dtype},
            {},
            pessimistic_nulls=pessimistic,
        )

    assert_eq(df_expected, df_actual)


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

    with pytest.warns(FutureWarning):
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

    with pytest.warns(FutureWarning):
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
