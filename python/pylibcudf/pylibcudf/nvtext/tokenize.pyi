# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

class TokenizeVocabulary:
    def __init__(
        self,
        vocab: Column,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ): ...

def tokenize_scalar(
    input: Column,
    delimiter: Scalar | None = None,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def tokenize_column(
    input: Column,
    delimiters: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def count_tokens_scalar(
    input: Column,
    delimiter: Scalar | None = None,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def count_tokens_column(
    input: Column,
    delimiters: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def character_tokenize(
    input: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def detokenize(
    input: Column,
    row_indices: Column,
    separator: Scalar | None = None,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def tokenize_with_vocabulary(
    input: Column,
    vocabulary: TokenizeVocabulary,
    delimiter: Scalar,
    default_id: int = -1,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
