# Copyright (c) 2018, NVIDIA CORPORATION.

from itertools import product

import pytest
import numpy as np

from . import utils

from cudf import Series
from math import floor


@pytest.mark.parametrize('nelem,masked',
                         list(product([2, 10, 100, 1000],
                                      [True, False])))
def test_applymap_round(nelem, masked):
    # Generate data
    np.random.seed(0)
    data = np.random.random(nelem) * 100

    if masked:
        # Make mask
        bitmask = utils.random_bitmask(nelem)
        boolmask = np.asarray(utils.expand_bits_to_bytes(bitmask),
                              dtype=np.bool)[:nelem]
        data[~boolmask] = np.nan

    sr = Series(data)

    if masked:
        # Mask the Series
        sr = sr.set_mask(bitmask)

    # Call applymap
    out = sr.applymap(lambda x: (floor(x) + 1 if x - floor(x) >= 0.5
                                 else floor(x)))

    if masked:
        # Fill masked values
        out = out.fillna(np.nan)

    # Check
    expect = np.round(data)
    got = out.to_array()
    np.testing.assert_array_almost_equal(expect, got)
