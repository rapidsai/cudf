# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.strings.regex_flags import RegexFlags

class RegexProgram:
    def __init__(self): ...
    @staticmethod
    def create(pattern: str, flags: RegexFlags) -> RegexProgram: ...
