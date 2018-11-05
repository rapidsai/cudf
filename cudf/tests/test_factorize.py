# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest
import numpy as np
from cudf.dataframe import DataFrame, Series


@pytest.mark.parametrize('ncats,nelem',
                         [(2, 2), (2, 10), (10, 100)])
def test_factorize(ncats, nelem):
    df = DataFrame()
    np.random.seed(0)

    # initialize data frame
    df['cats'] = arr = np.random.randint(2, size=10, dtype=np.int32)

    uvals, labels = df['cats'].factorize()
    np.testing.assert_array_equal(labels.to_array(), sorted(set(arr)))
    assert isinstance(uvals, Series)
    assert isinstance(labels, Series)

    encoder = dict((v, i) for i, v in enumerate(labels))
    handcoded = [encoder[v] for v in arr]
    np.testing.assert_array_equal(uvals.to_array(), handcoded)
