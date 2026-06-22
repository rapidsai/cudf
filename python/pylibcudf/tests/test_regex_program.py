# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import pylibcudf as plc


@pytest.mark.parametrize("pat", ["(", "*", "\\"])
def test_regex_program_invalid(pat):
    with pytest.raises(RuntimeError):
        plc.strings.regex_program.RegexProgram.create(
            pat, plc.strings.regex_flags.RegexFlags.DEFAULT
        )
