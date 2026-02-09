# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


def format_bytes(nbytes: int) -> str:
    """Format `nbytes` to a human readable string"""
    n = float(nbytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024:
            if n.is_integer():
                return f"{int(n)}{unit}"
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f} PiB"
