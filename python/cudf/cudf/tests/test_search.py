import pytest
import cudf

from cudf.tests.utils import assert_eq, gen_rand, random_bitmask


@pytest.mark.parametrize('side', ['left', 'right'])
def test_searchsorted(side):
    nelem = 1000
    column_data = gen_rand('float64', nelem)
    column_mask = random_bitmask(nelem)

    values_data = gen_rand('float64', nelem)
    values_mask = random_bitmask(nelem)

    sr = cudf.Series.from_masked_array(column_data, column_mask)
    vals = cudf.Series.from_masked_array(values_data, values_mask)

    sr = sr.sort_values()

    psr = sr.to_pandas()
    pvals = vals.to_pandas()

    expect = psr.searchsorted(pvals, side)
    got = sr.searchsorted(vals, side)

    assert_eq(expect, got.to_array())


@pytest.mark.parametrize('side', ['left', 'right'])
def test_searchsorted_categorical(side):
    import pandas as pd

    cat1 = pd.Categorical(
        ["a", "a", "b", "c", "a"], categories=["a", "b", "c"], ordered=True
    )
    psr1 = pd.Series(cat1).sort_values()
    sr1 = cudf.Series(cat1).sort_values()
    cat2 = pd.Categorical(
        ["a", "b", "a", "c", "b"], categories=["a", "b", "c"], ordered=True
    )
    psr2 = pd.Series(cat2)
    sr2 = cudf.Series(cat2)

    expect = psr1.searchsorted(psr2, side)
    got = sr1.searchsorted(sr2, side)

    assert_eq(expect, got.to_array())


