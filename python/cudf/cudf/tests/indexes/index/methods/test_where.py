# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.mark.parametrize(
    "data,condition,other,error",
    [
        (pd.Index(range(5)), pd.Index(range(5)) > 0, None, None),
        (pd.Index([1, 2, 3]), pd.Index([1, 2, 3]) != 2, None, None),
        (pd.Index(list("abc")), pd.Index(list("abc")) == "c", None, None),
        (
            pd.Index(list("abc")),
            pd.Index(list("abc")) == "c",
            pd.Index(list("xyz")),
            None,
        ),
        (pd.Index(range(5)), pd.Index(range(4)) > 0, None, ValueError),
        (
            pd.Index(range(5)),
            pd.Index(range(5)) > 1,
            10,
            None,
        ),
        (
            pd.Index(np.arange(10)),
            (pd.Index(np.arange(10)) % 3) == 0,
            -pd.Index(np.arange(10)),
            None,
        ),
        (
            pd.Index([1, 2, np.nan]),
            pd.Index([1, 2, np.nan]) == 4,
            None,
            None,
        ),
        (
            pd.Index([1, 2, np.nan]),
            pd.Index([1, 2, np.nan]) != 4,
            None,
            None,
        ),
        (
            pd.Index([-2, 3, -4, -79]),
            [True, True, True],
            None,
            ValueError,
        ),
        (
            pd.Index([-2, 3, -4, -79]),
            [True, True, True, False],
            None,
            None,
        ),
        (
            pd.Index([-2, 3, -4, -79]),
            [True, True, True, False],
            17,
            None,
        ),
        (pd.Index(list("abcdgh")), pd.Index(list("abcdgh")) != "g", "3", None),
        (
            pd.Index(list("abcdgh")),
            pd.Index(list("abcdg")) != "g",
            "3",
            ValueError,
        ),
        (
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]) != "a",
            "a",
            None,
        ),
        (
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
            pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]) != "a",
            "b",
            None,
        ),
        (
            pd.MultiIndex.from_tuples(
                list(
                    zip(
                        *[
                            [
                                "bar",
                                "bar",
                                "baz",
                                "baz",
                                "foo",
                                "foo",
                                "qux",
                                "qux",
                            ],
                            [
                                "one",
                                "two",
                                "one",
                                "two",
                                "one",
                                "two",
                                "one",
                                "two",
                            ],
                        ],
                        strict=True,
                    )
                )
            ),
            pd.MultiIndex.from_tuples(
                list(
                    zip(
                        *[
                            [
                                "bar",
                                "bar",
                                "baz",
                                "baz",
                                "foo",
                                "foo",
                                "qux",
                                "qux",
                            ],
                            [
                                "one",
                                "two",
                                "one",
                                "two",
                                "one",
                                "two",
                                "one",
                                "two",
                            ],
                        ],
                        strict=True,
                    )
                )
            )
            != "a",
            None,
            NotImplementedError,
        ),
    ],
)
def test_index_where(data, condition, other, error):
    ps = data
    gs = cudf.from_pandas(data)

    ps_condition = condition
    if isinstance(condition, pd.Index):
        gs_condition = cudf.from_pandas(condition)
    else:
        gs_condition = condition

    ps_other = other
    if isinstance(condition, pd.Index):
        gs_other = cudf.from_pandas(other)
    else:
        gs_other = other

    if error is None:
        if hasattr(ps, "dtype") and isinstance(ps.dtype, pd.CategoricalDtype):
            expect = ps.where(ps_condition, other=ps_other)
            got = gs.where(gs_condition, other=gs_other)
            np.testing.assert_array_equal(
                expect.codes,
                got.codes.astype(expect.codes.dtype).fillna(-1).to_numpy(),
            )
            assert_eq(expect.categories, got.categories)
        else:
            assert_eq(
                ps.where(ps_condition, other=ps_other),
                gs.where(gs_condition, other=gs_other).to_pandas(),
            )
    else:
        assert_exceptions_equal(
            lfunc=ps.where,
            rfunc=gs.where,
            lfunc_args_and_kwargs=([ps_condition], {"other": ps_other}),
            rfunc_args_and_kwargs=([gs_condition], {"other": gs_other}),
        )
