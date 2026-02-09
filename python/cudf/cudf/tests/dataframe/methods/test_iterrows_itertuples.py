# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

import cudf


def test_dataframe_iterrows_itertuples():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with pytest.raises(
        TypeError,
        match=re.escape(
            "cuDF does not support iteration of DataFrame "
            "via itertuples. Consider using "
            "`.to_pandas().itertuples()` "
            "if you wish to iterate over namedtuples."
        ),
    ):
        df.itertuples()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "cuDF does not support iteration of DataFrame "
            "via iterrows. Consider using "
            "`.to_pandas().iterrows()` "
            "if you wish to iterate over each row."
        ),
    ):
        df.iterrows()
