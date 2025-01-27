# Copyright (c) 2024, NVIDIA CORPORATION.
import zoneinfo

import pytest

import pylibcudf as plc


def test_make_timezone_transition_table():
    if len(zoneinfo.TZPATH) == 0:
        pytest.skip("No TZPATH available.")
    tz_path = zoneinfo.TZPATH[0]
    result = plc.io.timezone.make_timezone_transition_table(
        tz_path, "America/Los_Angeles"
    )
    assert isinstance(result, plc.Table)
    assert result.num_rows() > 0
