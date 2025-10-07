# Copyright (c) 2024, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def edit_distance(
    input: Column, targets: Column, stream: Stream | None = None
) -> Column: ...
def edit_distance_matrix(
    input: Column, stream: Stream | None = None
) -> Column: ...
