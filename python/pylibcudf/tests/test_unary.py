# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pylibcudf as plc


def test_is_supported_cast():
    assert plc.unary.is_supported_cast(
        plc.DataType(plc.TypeId.INT8), plc.DataType(plc.TypeId.UINT64)
    )
    assert plc.unary.is_supported_cast(
        plc.DataType(plc.TypeId.DURATION_MILLISECONDS),
        plc.DataType(plc.TypeId.UINT64),
    )
    assert not plc.unary.is_supported_cast(
        plc.DataType(plc.TypeId.INT32), plc.DataType(plc.TypeId.TIMESTAMP_DAYS)
    )
    assert not plc.unary.is_supported_cast(
        plc.DataType(plc.TypeId.INT32), plc.DataType(plc.TypeId.STRING)
    )
