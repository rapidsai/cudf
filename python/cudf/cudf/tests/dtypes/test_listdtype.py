# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import cudf
from cudf.utils.dtypes import cudf_dtype_to_pa_type


def test_listdtype_hash():
    a = cudf.ListDtype("int64")
    b = cudf.ListDtype("int64")

    assert hash(a) == hash(b)

    c = cudf.ListDtype("int32")

    assert hash(a) != hash(c)


def test_list_dtype_pyarrow_round_trip(all_supported_types_as_str, request):
    request.applymarker(
        pytest.mark.xfail(
            all_supported_types_as_str == "category",
            reason=f"{all_supported_types_as_str} conversion to Arrow not implemented",
        )
    )
    pa_type = pa.list_(
        cudf_dtype_to_pa_type(cudf.dtype(all_supported_types_as_str))
    )
    expect = pa_type
    got = cudf.ListDtype.from_arrow(expect).to_arrow()
    assert expect.equals(got)


def test_list_dtype_eq():
    lhs = cudf.ListDtype("int32")
    rhs = cudf.ListDtype("int32")
    assert lhs == rhs
    rhs = cudf.ListDtype("int64")
    assert lhs != rhs


def test_list_nested_dtype():
    dt = cudf.ListDtype(cudf.ListDtype("int32"))
    expect = cudf.ListDtype("int32")
    got = dt.element_type
    assert expect == got
