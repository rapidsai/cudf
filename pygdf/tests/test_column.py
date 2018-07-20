from itertools import product

import pytest
import numpy as np

from . import utils

from pygdf.column import Column
from pygdf.buffer import Buffer


@pytest.mark.parametrize('nelem', [0, 1, 2, 10, 100, 1000])
def test_column_from_cffi_view_data_only(nelem):
    # Generate data
    np.random.seed(0)
    data = np.random.random(nelem)

    # Make a Column.cffi_view::gdf_column
    col = Column(data=Buffer(data))
    gdf_column_struct = col.cffi_view

    got = Column.from_cffi_view(gdf_column_struct)
    # Check full array equal
    np.testing.assert_array_equal(got.to_array(), data)
    # Check first half equal
    assert len(got[:nelem//2]) == nelem // 2
    np.testing.assert_array_equal(got[:nelem//2].to_array(), data[:nelem//2])
    # Check latter half equal
    assert len(got[nelem//2:]) == nelem - nelem // 2
    np.testing.assert_array_equal(got[nelem//2:].to_array(), data[nelem//2:])
    # Check middle slice
    np.testing.assert_array_equal(got[nelem//4:nelem//2].to_array(),
                                  data[nelem//4:nelem//2:])


@pytest.mark.parametrize('nelem', [10, 100, 1000])
def test_column_from_cffi_view_masked(nelem):
    # Generate data
    np.random.seed(0)
    data = np.random.random(nelem)
    # Make masked column
    bitmask = utils.random_bitmask(nelem)
    boolmask = np.asarray(utils.expand_bits_to_bytes(bitmask),
                          dtype=np.bool)[:nelem]
    col = Column(Buffer(data)).set_mask(bitmask)
    assert col.has_null_mask
    # Make a Column.cffi_view::gdf_column
    gdf_column_struct = col.cffi_view
    got = Column.from_cffi_view(gdf_column_struct)
    # Check full array equal
    np.testing.assert_array_equal(got.to_array(), data[boolmask])
