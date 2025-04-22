# Copyright (c) 2025, NVIDIA CORPORATION.
import pylibcudf as plc


def test_is_ptds_enabled():
    assert isinstance(plc.utilities.is_ptds_enabled(), bool)
