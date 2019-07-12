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
