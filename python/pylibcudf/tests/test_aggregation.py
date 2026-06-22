# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pylibcudf as plc


def test_repr_name():
    assert repr(plc.aggregation.any()) == "<Aggregation(<Kind.ANY: 7>)>"
