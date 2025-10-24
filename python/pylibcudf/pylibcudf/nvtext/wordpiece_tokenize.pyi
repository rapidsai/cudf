# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

class WordPieceVocabulary:
    def __init__(
        self,
        vocab: Column,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ): ...

def wordpiece_tokenize(
    input: Column,
    vocabulary: WordPieceVocabulary,
    max_words_per_row: int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
