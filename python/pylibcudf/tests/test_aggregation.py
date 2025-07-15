# Copyright (c) 2025, NVIDIA CORPORATION.

import pylibcudf as plc


def test_str_name():
    assert str(plc.aggregation.any()) == "Aggregation(kind=Kind.ANY)"
