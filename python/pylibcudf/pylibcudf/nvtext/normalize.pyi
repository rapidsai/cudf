# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

class CharacterNormalizer:
    def __init__(
        self,
        do_lower_case: bool,
        special_tokens: Column,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ): ...

def normalize_spaces(
    input: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def normalize_characters(
    input: Column,
    normalizer: CharacterNormalizer,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
