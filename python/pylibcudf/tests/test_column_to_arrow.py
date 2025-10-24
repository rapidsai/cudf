# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from utils import assert_column_eq


def test_column_to_arrow(table_data):
    plc_tbl, _ = table_data
    for col in plc_tbl.tbl.columns():
        assert_column_eq(col, col.to_arrow())
