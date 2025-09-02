# Copyright (c) 2025, NVIDIA CORPORATION.

import pylibcudf as plc


def test_repr_name():
    assert repr(plc.aggregation.any()) == "<Aggregation(<Kind.ANY: 7>)>"
