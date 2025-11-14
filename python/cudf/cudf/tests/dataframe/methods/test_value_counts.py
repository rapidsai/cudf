# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest

import cudf
from cudf.testing import assert_eq


def test_value_counts_no_subset():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [1, 1, 0]})
    with pytest.raises(KeyError):
        gdf.value_counts(subset=["not_a_column_name"])


@pytest.mark.parametrize(
    "data",
    [
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        {
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", None, None, "Louise"],
        },
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("use_subset", [True, False])
def test_value_counts(
    data,
    sort,
    ascending,
    normalize,
    dropna,
    use_subset,
):
    subset = [next(iter(data.keys()))]
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    got = gdf.value_counts(
        subset=subset if (use_subset) else None,
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        dropna=dropna,
    )
    expected = pdf.value_counts(
        subset=subset if (use_subset) else None,
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        dropna=dropna,
    )

    if not dropna:
        # Convert the Pandas series to a cuDF one due to difference
        # in the handling of NaNs between the two (<NA> in cuDF and
        # NaN in Pandas) when dropna=False.
        assert_eq(got.sort_index(), cudf.from_pandas(expected).sort_index())
    else:
        assert_eq(got.sort_index(), expected.sort_index())
