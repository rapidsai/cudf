# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Final

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table

LIBCUDF_DEFAULT_HASH_SEED: Final[int]

def murmurhash3_x86_32(
    input: Table,
    seed: int = ...,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def murmurhash3_x64_128(
    input: Table,
    seed: int = ...,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def xxhash_32(
    input: Table,
    seed: int = ...,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def xxhash_64(
    input: Table,
    seed: int = ...,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def md5(
    input: Table,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def sha1(
    input: Table,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def sha224(
    input: Table,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def sha256(
    input: Table,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def sha384(
    input: Table,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def sha512(
    input: Table,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
