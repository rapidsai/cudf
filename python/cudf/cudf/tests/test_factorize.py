# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pytest

from cudf.core import DataFrame, Series
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize("ncats,nelem", [(2, 2), (2, 10), (10, 100)])
def test_factorize(ncats, nelem):
    df = DataFrame()
    np.random.seed(0)

    # initialize data frame
    df["cats"] = arr = np.random.randint(2, size=10, dtype=np.int32)

    uvals, labels = df["cats"].factorize()
    np.testing.assert_array_equal(labels.to_array(), sorted(set(arr)))
    assert isinstance(uvals, Series)
    assert isinstance(labels, Series)

    encoder = dict((labels[idx], idx) for idx in range(len(labels)))
    handcoded = [encoder[v] for v in arr]
    np.testing.assert_array_equal(uvals.to_array(), handcoded)


def test_factorize_index():
    df = DataFrame()
    df["col1"] = ["C", "H", "C", "W", "W", "W", "W", "W", "C", "W"]
    df["col2"] = [
        2992443.0,
        2992447.0,
        2992466.0,
        2992440.0,
        2992441.0,
        2992442.0,
        2992444.0,
        2992445.0,
        2992446.0,
        2992448.0,
    ]

    assert_eq(
        df.col1.factorize()[0].to_array(), df.to_pandas().col1.factorize()[0]
    )
    assert_eq(
        df.col1.factorize()[1].to_pandas().values,
        df.to_pandas().col1.factorize()[1].values,
    )

    df = df.set_index("col2")

    assert_eq(
        df.col1.factorize()[0].to_array(), df.to_pandas().col1.factorize()[0]
    )
    assert_eq(
        df.col1.factorize()[1].to_pandas().values,
        df.to_pandas().col1.factorize()[1].values,
    )
