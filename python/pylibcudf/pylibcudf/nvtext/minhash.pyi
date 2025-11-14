# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def minhash(
    input: Column,
    seed: int | np.unsignedinteger[Any],
    a: Column,
    b: Column,
    width: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def minhash64(
    input: Column,
    seed: int | np.unsignedinteger[Any],
    a: Column,
    b: Column,
    width: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def minhash_ngrams(
    input: Column,
    ngrams: int,
    seed: int | np.unsignedinteger[Any],
    a: Column,
    b: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def minhash64_ngrams(
    input: Column,
    ngrams: int,
    seed: int | np.unsignedinteger[Any],
    a: Column,
    b: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
