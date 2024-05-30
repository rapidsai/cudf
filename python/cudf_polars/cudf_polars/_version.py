# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.resources

__version__ = (
    importlib.resources.files(__package__).joinpath("VERSION").read_text().strip()
)
try:
    __git_commit__ = (
        importlib.resources.files(__package__)
        .joinpath("GIT_COMMIT")
        .read_text()
        .strip()
    )
except FileNotFoundError:
    __git_commit__ = ""

__all__ = ["__git_commit__", "__version__"]
