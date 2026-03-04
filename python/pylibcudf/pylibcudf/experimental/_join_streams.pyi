# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.stream import Stream

def join_streams(streams: list[Stream], stream: Stream) -> None: ...
