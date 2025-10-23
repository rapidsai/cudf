# Copyright (c) 2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

def join_streams(streams: list[Stream], stream: Stream) -> None: ...
