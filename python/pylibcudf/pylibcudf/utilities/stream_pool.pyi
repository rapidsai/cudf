# Copyright (c) 2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

def join_streams(streams: list[Stream], stream: Stream) -> None:
    """
    Synchronize a stream to an event on a set of streams.

    After joining the streams, data that is valid on any of the
    streams in ``streams`` is also valid on ``stream``.

    Parameters
    ----------
    streams
        Streams to wait on.
    stream
        Joined stream that synchronizes with the waited-on streams.
    """
