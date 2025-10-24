# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import overload

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.regex_flags import RegexFlags
from pylibcudf.strings.regex_program import RegexProgram

@overload
def replace_re(
    input: Column,
    pattern: RegexProgram,
    replacement: Scalar,
    max_replace_count: int = -1,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
@overload
def replace_re(
    input: Column,
    patterns: list[str],
    replacement: Column,
    max_replace_count: int = -1,
    flags: RegexFlags = RegexFlags.DEFAULT,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def replace_with_backrefs(
    input: Column,
    prog: RegexProgram,
    replacement: str,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
