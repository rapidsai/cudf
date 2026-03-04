# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf


def test_to_records_noindex():
    aa = np.arange(10, dtype=np.int32)
    bb = np.arange(10, 20, dtype=np.float64)
    df = cudf.DataFrame(
        {
            "a": aa,
            "b": bb,
        }
    )

    rec = df.to_records(index=False)
    assert rec.dtype.names == ("a", "b")
    np.testing.assert_array_equal(rec["a"], aa)
    np.testing.assert_array_equal(rec["b"], bb)


def test_to_records_withindex():
    aa = np.arange(10, dtype=np.int32)
    bb = np.arange(10, 20, dtype=np.float64)
    df = cudf.DataFrame(
        {
            "a": aa,
            "b": bb,
        }
    )

    rec_indexed = df.to_records(index=True)
    assert rec_indexed.size == len(aa)
    assert rec_indexed.dtype.names == ("index", "a", "b")
    np.testing.assert_array_equal(rec_indexed["a"], aa)
    np.testing.assert_array_equal(rec_indexed["b"], bb)
    np.testing.assert_array_equal(rec_indexed["index"], np.arange(10))
