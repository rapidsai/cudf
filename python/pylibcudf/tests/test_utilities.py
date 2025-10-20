# Copyright (c) 2025, NVIDIA CORPORATION.
import pytest

from rmm.pylibrmm.stream import Stream

import pylibcudf as plc


def test_is_ptds_enabled():
    assert isinstance(plc.utilities.is_ptds_enabled(), bool)


def test_join_streams_basic():
    """Test basic join_streams functionality."""
    # Create multiple streams
    stream1 = Stream()
    stream2 = Stream()
    main_stream = Stream()

    # join_streams should synchronize main_stream with stream1 and stream2
    plc.utilities.stream_pool.join_streams([stream1, stream2], main_stream)

    # The function should complete without error
    # We can't easily verify the synchronization directly,
    # but we can check that it doesn't raise an exception


def test_join_streams_empty_list():
    """Test join_streams with empty list of streams."""
    main_stream = Stream()

    # Should handle empty list gracefully
    plc.utilities.stream_pool.join_streams([], main_stream)


def test_join_streams_single_stream():
    """Test join_streams with a single stream."""
    stream1 = Stream()
    main_stream = Stream()

    plc.utilities.stream_pool.join_streams([stream1], main_stream)


def test_join_streams_multiple_streams():
    """Test join_streams with many streams."""
    streams = [Stream() for _ in range(10)]
    main_stream = Stream()

    plc.utilities.stream_pool.join_streams(streams, main_stream)


def test_join_streams_type_error():
    """Test that join_streams raises appropriate errors for invalid inputs."""
    main_stream = Stream()

    # Test with non-list input
    with pytest.raises(TypeError):
        plc.utilities.stream_pool.join_streams(None, main_stream)

    # Test with non-Stream in list
    with pytest.raises(TypeError):
        plc.utilities.stream_pool.join_streams([None], main_stream)

    # Test with non-Stream as main stream
    with pytest.raises(TypeError):
        plc.utilities.stream_pool.join_streams([Stream()], None)
