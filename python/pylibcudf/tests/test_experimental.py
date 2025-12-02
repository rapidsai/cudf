# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest

from rmm.pylibrmm.stream import Stream

import pylibcudf as plc


@pytest.mark.parametrize(
    "streams, stream",
    [
        ([], Stream()),
        ([Stream()], Stream()),
        ([Stream() for _ in range(10)], Stream()),
    ],
)
def test_join_streams(streams: list[Stream], stream: Stream):
    # Just check that there isn't an error / segfault.
    # We can't easily verify correctness with this type of test.
    plc.experimental.join_streams(streams, stream)


def test_join_streams_type_error():
    """Test that join_streams raises appropriate errors for invalid inputs."""
    main_stream = Stream()

    # Test with non-list input
    with pytest.raises(TypeError):
        plc.experimental.join_streams(None, main_stream)

    # Test with non-Stream in list
    with pytest.raises(
        TypeError,
        match="Cannot convert NoneType to rmm.pylibrmm.stream.Stream",
    ):
        plc.experimental.join_streams([None], main_stream)

    # Test with non-Stream as main stream
    with pytest.raises(
        TypeError,
        match="Cannot convert NoneType to rmm.pylibrmm.stream.Stream",
    ):
        plc.experimental.join_streams([Stream()], None)
