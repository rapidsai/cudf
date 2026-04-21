# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import zoneinfo

import pytest

import pylibcudf as plc


def _first_available_tzpath_with(zone_name: str) -> str | None:
    """Return the first directory in ``zoneinfo.TZPATH`` that contains
    ``zone_name`` as a TZif file, or ``None`` if no such directory exists.
    """
    for tz_path in zoneinfo.TZPATH:
        if os.path.isfile(os.path.join(tz_path, zone_name)):
            return tz_path
    return None


def test_make_timezone_transition_table():
    tz_path = _first_available_tzpath_with("America/Los_Angeles")
    if tz_path is None:
        pytest.skip("No TZif directory with America/Los_Angeles available.")
    result = plc.io.timezone.make_timezone_transition_table(
        tz_path, "America/Los_Angeles"
    )
    assert isinstance(result, plc.Table)
    assert result.num_rows() > 0


def test_make_tz_transition_table_resolves_backward_alias(tmp_path):
    src_dir = _first_available_tzpath_with("America/Los_Angeles")
    if src_dir is None:
        pytest.skip("No TZif directory with America/Los_Angeles available.")

    os.makedirs(tmp_path / "America", exist_ok=True)
    shutil.copyfile(
        os.path.join(src_dir, "America", "Los_Angeles"),
        tmp_path / "America" / "Los_Angeles",
    )
    (tmp_path / "tzdata.zi").write_text("L America/Los_Angeles US/Pacific\n")

    canonical = plc.io.timezone.make_timezone_transition_table(
        str(tmp_path), "America/Los_Angeles"
    )
    via_alias = plc.io.timezone.make_timezone_transition_table(
        str(tmp_path), "US/Pacific"
    )
    assert canonical.num_rows() == via_alias.num_rows()
    assert canonical.num_columns() == via_alias.num_columns()
