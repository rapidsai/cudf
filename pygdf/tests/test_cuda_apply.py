"""
Test method that apply GPU kernel to a frame.
"""

import pytest
import numpy as np

from pygdf import DataFrame


@pytest.mark.parametrize('nelem', [1, 2, 64, 128, 1000, 5000])
def test_df_apply_rows(nelem):
    def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        for i, (x, y, z) in enumerate(zip(in1, in2, in3)):
            out1[i] = extra2 * x - extra1 * y
            out2[i] = y - extra1 * z

    df = DataFrame()
    df['in1'] = in1 = np.arange(nelem)
    df['in2'] = in2 = np.arange(nelem)
    df['in3'] = in3 = np.arange(nelem)

    extra1 = 2.3
    extra2 = 3.4

    expect_out1 = extra2 * in1 - extra1 * in2
    expect_out2 = in2 - extra1 * in3

    outdf = df.apply_rows(kernel,
                          incols=['in1', 'in2', 'in3'],
                          outcols=dict(out1=np.float64, out2=np.float64),
                          kwargs=dict(extra1=extra1, extra2=extra2))

    got_out1 = outdf['out1'].to_array()
    got_out2 = outdf['out2'].to_array()

    np.testing.assert_array_almost_equal(got_out1, expect_out1)
    np.testing.assert_array_almost_equal(got_out2, expect_out2)


@pytest.mark.parametrize('nelem', [1, 2, 64, 128, 1000, 5000])
@pytest.mark.parametrize('chunksize', [1, 2, 3, 4, 23])
def test_df_apply_chunks(nelem, chunksize):
    def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        for i, (x, y, z) in enumerate(zip(in1, in2, in3)):
            out1[i] = extra2 * x - extra1 * y + z
            out2[i] = i

    df = DataFrame()
    df['in1'] = in1 = np.arange(nelem)
    df['in2'] = in2 = np.arange(nelem)
    df['in3'] = in3 = np.arange(nelem)

    extra1 = 2.3
    extra2 = 3.4

    expect_out1 = extra2 * in1 - extra1 * in2 + in3
    expect_out2 = np.arange(len(df)) % chunksize

    outdf = df.apply_chunks(kernel,
                            incols=['in1', 'in2', 'in3'],
                            outcols=dict(out1=np.float64, out2=np.int32),
                            kwargs=dict(extra1=extra1, extra2=extra2),
                            chunks=chunksize)

    got_out1 = outdf['out1']
    got_out2 = outdf['out2']

    np.testing.assert_array_almost_equal(got_out1, expect_out1)
    np.testing.assert_array_almost_equal(got_out2, expect_out2)

