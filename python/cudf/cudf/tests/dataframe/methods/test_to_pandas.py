# Copyright (c) 2025, NVIDIA CORPORATION.


import pytest

import cudf


def test_list_to_pandas_nullable_true():
    df = cudf.DataFrame({"a": cudf.Series([[1, 2, 3]])})
    with pytest.raises(NotImplementedError):
        df.to_pandas(nullable=True)
