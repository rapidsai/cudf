# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(
            {
                "num_legs": [2, 4],
                "num_wings": [2, 0],
                "bird_cats": pd.Series(
                    ["sparrow", "pigeon"],
                    dtype="category",
                    index=["falcon", "dog"],
                ),
            },
            index=["falcon", "dog"],
        ),
        pd.DataFrame(
            {"num_legs": [8, 2], "num_wings": [0, 2]},
            index=["spider", "falcon"],
        ),
        pd.DataFrame(
            {
                "num_legs": [8, 2, 1, 0, 2, 4, 5],
                "num_wings": [2, 0, 2, 1, 2, 4, -1],
            }
        ),
        pd.DataFrame({"a": ["a", "b", "c"]}, dtype="category"),
        pd.DataFrame({"a": ["a", "b", "c"]}),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [0, 2],
        {"num_wings": [0, 3]},
        pd.DataFrame(
            {"num_legs": [8, 2], "num_wings": [0, 2]},
            index=["spider", "falcon"],
        ),
        pd.DataFrame(
            {
                "num_legs": [2, 4],
                "num_wings": [2, 0],
                "bird_cats": pd.Series(
                    ["sparrow", "pigeon"],
                    dtype="category",
                    index=["falcon", "dog"],
                ),
            },
            index=["falcon", "dog"],
        ),
        ["sparrow", "pigeon"],
        pd.Series(["sparrow", "pigeon"], dtype="category"),
        pd.Series([1, 2, 3, 4, 5]),
        "abc",
        123,
        pd.Series(["a", "b", "c"]),
        pd.Series(["a", "b", "c"], dtype="category"),
        pd.DataFrame({"a": ["a", "b", "c"]}, dtype="category"),
    ],
)
def test_isin_dataframe(data, values):
    pdf = data
    gdf = cudf.from_pandas(pdf)

    if cudf.api.types.is_scalar(values):
        assert_exceptions_equal(
            lfunc=pdf.isin,
            rfunc=gdf.isin,
            lfunc_args_and_kwargs=([values],),
            rfunc_args_and_kwargs=([values],),
        )
    else:
        try:
            expected = pdf.isin(values)
        except TypeError as e:
            # Can't do isin with different categories
            if str(e) == (
                "Categoricals can only be compared if 'categories' "
                "are the same."
            ):
                return

        if isinstance(values, (pd.DataFrame, pd.Series)):
            values = cudf.from_pandas(values)

        got = gdf.isin(values)
        assert_eq(got, expected)


def test_isin_axis_duplicated_error():
    df = cudf.DataFrame(range(2))
    with pytest.raises(ValueError):
        df.isin(cudf.Series(range(2), index=[1, 1]))

    with pytest.raises(ValueError):
        df.isin(cudf.DataFrame(range(2), index=[1, 1]))

    with pytest.raises(ValueError):
        df.isin(cudf.DataFrame([[1, 2]], columns=[1, 1]))
