# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf


def test_to_pandas():
    df = cudf.DataFrame(
        {
            "a": np.arange(5, dtype=np.int32),
            "b": np.arange(10, 15, dtype=np.float64),
            "c": np.array([True, False, None, True, True]),
        }
    )

    pdf = df.to_pandas()

    assert tuple(df.columns) == tuple(pdf.columns)

    assert df["a"].dtype == pdf["a"].dtype
    assert df["b"].dtype == pdf["b"].dtype

    # Notice, the dtype differ when Pandas and cudf boolean series
    # contains None/NaN
    assert df["c"].dtype == np.bool_
    assert pdf["c"].dtype == np.object_

    assert len(df["a"]) == len(pdf["a"])
    assert len(df["b"]) == len(pdf["b"])
    assert len(df["c"]) == len(pdf["c"])


def test_list_to_pandas_nullable_true():
    df = cudf.DataFrame({"a": cudf.Series([[1, 2, 3]])})
    with pytest.raises(NotImplementedError):
        df.to_pandas(nullable=True)
