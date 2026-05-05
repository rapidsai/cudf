# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.utils import CudaStreamLike

class CharacterNormalizer:
    def __init__(
        self,
        do_lower_case: bool,
        special_tokens: Column,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ): ...

def normalize_spaces(
    input: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def normalize_characters(
    input: Column,
    normalizer: CharacterNormalizer,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
