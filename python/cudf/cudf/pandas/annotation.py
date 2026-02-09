# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

try:
    import nvtx
except ImportError:

    class nvtx:  # type: ignore[no-redef]
        """Noop-stub with the same API as nvtx."""

        push_range = lambda *args, **kwargs: None  # noqa: E731
        pop_range = lambda *args, **kwargs: None  # noqa: E731

        class annotate:
            """No-op annotation/context-manager"""

            def __init__(
                self,
                message: str | None = None,
                color: str | None = None,
                domain: str | None = None,
                category: str | int | None = None,
            ):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            __call__ = lambda self, fn: fn  # noqa: E731
