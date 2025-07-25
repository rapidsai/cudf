# Copyright (c) 2025, NVIDIA CORPORATION.


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("codes", [[0, 1, 2], [-1, 0, 1]])
def test_multiindex_to_frame(codes):
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
        ],
        [
            codes,
        ],
    )
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex.to_frame(), gdfIndex.to_frame())
