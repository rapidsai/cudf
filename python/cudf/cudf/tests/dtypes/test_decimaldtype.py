# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf


@pytest.mark.parametrize(
    "decimal_type",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype],
)
def test_decimal_dtype_arrow_roundtrip(decimal_type):
    dt = decimal_type(4, 2)
    pa_type = decimal_type.PA_TYPE(4, 2)
    assert dt.to_arrow() == pa_type
    assert dt == decimal_type.from_arrow(pa_type)


@pytest.mark.parametrize(
    "decimal_type,max_precision",
    [
        (cudf.Decimal32Dtype, 9),
        (cudf.Decimal64Dtype, 18),
        (cudf.Decimal128Dtype, 38),
    ],
)
def test_max_precision(decimal_type, max_precision):
    decimal_type(scale=0, precision=max_precision)
    with pytest.raises(ValueError):
        decimal_type(scale=0, precision=max_precision + 1)
