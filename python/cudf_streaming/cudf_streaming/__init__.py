# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""cuDF Streaming library."""

# If libcudf_streaming was installed as a wheel, request it to load the library
# symbols. Otherwise, assume the library is on a system path that ld can find.
try:
    import libcudf_streaming
except ModuleNotFoundError:
    pass
else:
    libcudf_streaming.load_library()
    del libcudf_streaming
