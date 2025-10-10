# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.strings.regex_flags import RegexFlags

class RegexProgram:
    def __init__(self): ...
    @staticmethod
    def create(pattern: str, flags: RegexFlags) -> RegexProgram: ...
