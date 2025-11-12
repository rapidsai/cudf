# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pylibcudf as plc


def test_is_ptds_enabled():
    assert isinstance(plc.utilities.is_ptds_enabled(), bool)
