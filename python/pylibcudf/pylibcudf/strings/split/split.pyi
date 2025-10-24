# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.regex_program import RegexProgram
from pylibcudf.table import Table

def split(
    strings_column: Column,
    delimiter: Scalar,
    maxsplit: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def rsplit(
    strings_column: Column,
    delimiter: Scalar,
    maxsplit: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def split_record(
    strings: Column,
    delimiter: Scalar,
    maxsplit: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def rsplit_record(
    strings: Column,
    delimiter: Scalar,
    maxsplit: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def split_re(
    input: Column,
    prog: RegexProgram,
    maxsplit: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def rsplit_re(
    input: Column,
    prog: RegexProgram,
    maxsplit: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def split_record_re(
    input: Column,
    prog: RegexProgram,
    maxsplit: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def rsplit_record_re(
    input: Column,
    prog: RegexProgram,
    maxsplit: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
