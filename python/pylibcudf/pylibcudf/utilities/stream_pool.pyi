# Copyright (c) 2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

def join_streams(streams: list[Stream], stream: Stream) -> None:
    """
    Synchronize a stream to an event on a set of streams.

    For details, see :cpp:func:`join_streams`.

    Parameters
    ----------
    streams : list[Stream]
        Streams to wait on.
    stream : Stream
        Joined stream that synchronizes with the waited-on streams.
    """
