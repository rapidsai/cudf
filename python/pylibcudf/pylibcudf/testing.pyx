# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.testing cimport get_default_stream

from rmm.pylibrmm.stream cimport Stream

__all__ = [
    "get_default_testing_stream",
]


def get_default_testing_stream():
    return Stream._from_cudaStream_t(get_default_stream().value())
