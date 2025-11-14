# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


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


@pytest.mark.parametrize(
    "pidx",
    [
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
            names=["a", "b", "c"],
        ),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
            names=["a", "a", "a"],
        ),
        pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], [5, 6, 7, 10], [11, 12, 12, 13]],
        ),
    ],
)
@pytest.mark.parametrize(
    "name", [None, no_default, ["x", "y", "z"], ["rapids", "rapids", "rapids"]]
)
@pytest.mark.parametrize("allow_duplicates", [True, False])
@pytest.mark.parametrize("index", [True, False])
def test_multiindex_to_frame_allow_duplicates(
    pidx, name, allow_duplicates, index
):
    gidx = cudf.from_pandas(pidx)

    if name is None or (
        (
            len(pidx.names) != len(set(pidx.names))
            and not all(x is None for x in pidx.names)
        )
        and not allow_duplicates
        and name is no_default
    ):
        assert_exceptions_equal(
            pidx.to_frame,
            gidx.to_frame,
            lfunc_args_and_kwargs=(
                [],
                {
                    "index": index,
                    "name": name,
                    "allow_duplicates": allow_duplicates,
                },
            ),
            rfunc_args_and_kwargs=(
                [],
                {
                    "index": index,
                    "name": name,
                    "allow_duplicates": allow_duplicates,
                },
            ),
        )
    else:
        if (
            len(pidx.names) != len(set(pidx.names))
            and not all(x is None for x in pidx.names)
            and not isinstance(name, list)
        ) or (isinstance(name, list) and len(name) != len(set(name))):
            # cudf doesn't have the ability to construct dataframes
            # with duplicate column names
            with pytest.raises(ValueError):
                gidx.to_frame(
                    index=index,
                    name=name,
                    allow_duplicates=allow_duplicates,
                )
        else:
            expected = pidx.to_frame(
                index=index, name=name, allow_duplicates=allow_duplicates
            )
            actual = gidx.to_frame(
                index=index, name=name, allow_duplicates=allow_duplicates
            )

            assert_eq(expected, actual)
