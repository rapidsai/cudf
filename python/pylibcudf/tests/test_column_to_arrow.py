# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest
from utils import assert_column_eq

from rmm.pylibrmm.stream import Stream


@pytest.mark.parametrize("stream", [None, Stream()])
def test_column_to_arrow(table_data, stream):
    plc_tbl, _ = table_data
    for col in plc_tbl.tbl.columns():
        assert_column_eq(col, col.to_arrow(stream=stream))
