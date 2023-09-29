# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

try:
    import nvtx
except ImportError:

    class nvtx:  # type: ignore
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
